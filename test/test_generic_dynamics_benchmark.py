from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path


import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.time_dynamics.tables import generic_dynamics_benchmark as bench
from pipelines.time_dynamics.tables import generic_dynamics_rows as rows_mod
from pipelines.time_dynamics.benchmarks import legacy_native
from pipelines.time_dynamics.benchmarks import registry as benchmark_registry
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DynamicsBenchmarkCase,
    DynamicsBenchmarkRow,
    DynamicsTableFields,
)
from pipelines.time_dynamics.tables.generic_dynamics_cases import get_generic_dynamics_case
from pipelines.time_dynamics.tables.table_lock_contract import with_class_settings_lock_manifest
from pipelines.scaffold.runtime_contract import CandidatePoolSource
from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERIC_DYNAMICS_MODULE_PATHS = (
    REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "dynamics_benchmark_contract.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "generic_dynamics_cases.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "generic_dynamics_rows.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "generic_dynamics_tables.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "generic_dynamics_ablation_matrix.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "generic_dynamics_benchmark.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "benchmarks" / "common.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "benchmarks" / "registry.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "benchmarks" / "legacy_native.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "benchmarks" / "qiskit_native.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "runners" / "generic_from_adapt_artifact.py",
    REPO_ROOT / "pipelines" / "time_dynamics" / "ap_mclachlan" / "support_patch.py",
)


def test_generic_dynamics_slice_has_no_qiskit_or_hh_comparator_imports() -> None:
    for path in GENERIC_DYNAMICS_MODULE_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "import qiskit" not in text
        assert "from qiskit" not in text
    rows_text = (
        REPO_ROOT / "pipelines" / "time_dynamics" / "tables" / "generic_dynamics_rows.py"
    ).read_text(encoding="utf-8")
    assert "hh_fixed_pvqd_benchmark" not in rows_text
    assert "hh_avqds_benchmark" not in rows_text
    for path in GENERIC_DYNAMICS_MODULE_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "external_parity_refs" not in text
        assert "pipelines.exact_bench.external_dynamics" not in text


def test_isolated_registry_wires_adaptive_pvqd_and_avqds_t_but_not_vff() -> None:
    assert benchmark_registry.supports_isolated_benchmark("dyn_exact_reference") is True
    assert benchmark_registry.runner_module_for_algorithm("dyn_exact_reference") == (
        "pipelines.time_dynamics.benchmarks.legacy_native"
    )
    assert benchmark_registry.supports_isolated_benchmark("dyn_adaptive_pvqd") is True
    assert benchmark_registry.runner_module_for_algorithm("dyn_adaptive_pvqd") == (
        "pipelines.time_dynamics.benchmarks.legacy_native"
    )
    assert benchmark_registry.supports_isolated_benchmark("dyn_avqds_t") is True
    assert benchmark_registry.runner_module_for_algorithm("dyn_avqds_t") == (
        "pipelines.time_dynamics.benchmarks.legacy_native"
    )
    assert benchmark_registry.supports_isolated_benchmark("dyn_qiskit_trotter_qrte") is True
    assert benchmark_registry.runner_module_for_algorithm("dyn_qiskit_trotter_qrte") == (
        "pipelines.time_dynamics.benchmarks.qiskit_native"
    )
    assert benchmark_registry.dispatch_label("dyn_qiskit_trotter_qrte") == (
        "generic_qiskit_community_comparator"
    )
    assert benchmark_registry.supports_isolated_benchmark("dyn_vff_like") is False


def _complex_payload(state: np.ndarray) -> list[dict[str, float]]:
    arr = np.asarray(state, dtype=complex).reshape(-1)
    return [{"re": float(z.real), "im": float(z.imag)} for z in arr]



def _fake_baseline_geometry(*, theta_dot_l2: float, runtime_parameter_count: int = 1) -> dict:
    return {
        "energy": 1.0,
        "variance": 0.5,
        "epsilon_proj_sq": 0.1,
        "epsilon_step_sq": 0.01,
        "rho_miss": 0.2,
        "rho_real": 0.8,
        "rho_num": 0.1,
        "step_objective_value": 0.3,
        "step_gain_ratio": 0.6,
        "theta_dot_l2": float(theta_dot_l2),
        "matrix_rank": int(runtime_parameter_count),
        "condition_number": 1.2,
        "regularization_lambda": 1.0e-8,
        "solve_mode": "pinv_reg",
        "logical_block_count": 1,
        "runtime_parameter_count": int(runtime_parameter_count),
        "planning_summary": {},
        "exact_cache_summary": {},
    }


def _fake_realtime_payload() -> dict:
    common_row = {
        "trajectory_sample_kind": "state_sample",
        "action_kind": "stay",
        "advances_time": True,
        "integrator_policy": "auto_euler_rk4",
        "integrator_used": "rk4",
        "integrator_auto_policy_schema": "auto_euler_rk4_policy_v1",
        "integrator_condition_number": 1.2,
        "integrator_condition_pass": True,
        "integrator_geometry_gate_pass": True,
        "integrator_euler_error_pass": True,
        "integrator_rho_miss_pass": True,
        "runtime_parameter_count_delta": 0,
        "rho_miss": 0.2,
        "rho_real": 0.8,
        "rho_num": 0.1,
        "epsilon_proj_sq": 0.1,
        "epsilon_step_sq": 0.01,
        "primary_density": 0.0,
        "primary_density_exact": 0.0,
        "site_occupations": [1.0],
        "site_occupations_exact": [1.0],
    }
    return {
        "summary": {
            "mode": "off",
            "controller_exact_input_mode": "benchmark_exact",
            "diagnostic_exact_reference_mode": "off",
            "decision_data_flow": "controller_off_exact_benchmark_diagnostic",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "max_abs_site_occupations_error": 0.25,
            "integrator_policy": "auto_euler_rk4",
        },
        "runtime_contract": {
            "controller_exact_input_mode": "benchmark_exact",
            "diagnostic_exact_reference_mode": "off",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
        },
        "compile_audit": {
            "selected_backend": {
                "compiled_count_2q": 10,
                "compiled_depth": 21,
                "compiled_size": 34,
                "backend_name": "unit_backend",
            }
        },
        "trajectory": [
            {
                **common_row,
                "checkpoint_index": 0,
                "time": 0.0,
                "energy_total": 10.0,
                "energy_total_controller": 10.0,
                "energy_total_exact": 1.0,
                "abs_energy_total_error": 9.0,
                "abs_primary_density_error": 0.1,
                "fidelity_exact": 1.0,
                "theta_dot_l2": 0.0,
                "theta_update_l2": 0.0,
                "baseline_geometry": _fake_baseline_geometry(theta_dot_l2=0.0),
            },
            {
                **common_row,
                "checkpoint_index": 1,
                "time": 0.2,
                "energy_total": 1.4,
                "energy_total_controller": 1.4,
                "energy_total_exact": 1.2,
                "abs_energy_total_error": 0.4,
                "abs_primary_density_error": 0.2,
                "fidelity_exact": 0.96,
                "theta_dot_l2": 0.5,
                "theta_update_l2": 0.1,
                "primary_density": 0.1,
                "primary_density_exact": 0.05,
                "site_occupations": [0.9],
                "site_occupations_exact": [0.95],
                "baseline_geometry": _fake_baseline_geometry(theta_dot_l2=0.5),
            },
        ],
        "ledger": [],
    }


def test_dynamics_manifest_promotes_generic_comparators_for_non_hh(tmp_path: Path) -> None:
    algorithms = (
        "dyn_exact_reference",
        "dyn_fixed_mclachlan",
        "dyn_product_formula_envelope",
        "dyn_qdrift",
        "dyn_fixed_pvqd",
        "dyn_adaptive_pvqd",
        "dyn_avqds",
        "dyn_avqds_t",
        "dyn_vff_like",
        "dyn_controller_full",
        "dyn_controller_ablation_matrix",
    )
    jobs = bench.build_dynamics_jobs(
        output_root=tmp_path,
        families=("hh", "hubbard"),
        algorithm_ids=algorithms,
        include_skipped=True,
    )

    hh = {job.algorithm_id: job for job in jobs if job.family == "hh"}
    hubbard = {job.algorithm_id: job for job in jobs if job.family == "hubbard"}

    assert hh["dyn_exact_reference"].status == "runnable"
    assert hh["dyn_product_formula_envelope"].status == "runnable"
    assert hh["dyn_vff_like"].status == "runnable"
    assert all(hh[alg].command for alg in ("dyn_exact_reference", "dyn_product_formula_envelope"))
    assert hh["dyn_exact_reference"].metadata["dispatch"] == "hh_legacy_wrapper"
    assert hh["dyn_controller_full"].status == "skipped_no_runner"
    assert not hh["dyn_controller_full"].command
    assert hh["dyn_controller_full"].metadata["dispatch"] == "hh_case_skipped"
    assert hh["dyn_controller_ablation_matrix"].status == "skipped_no_runner"
    assert not hh["dyn_controller_ablation_matrix"].command
    assert hh["dyn_controller_ablation_matrix"].metadata["dispatch"] == "hh_case_skipped"

    assert hubbard["dyn_exact_reference"].status == "runnable"
    assert hubbard["dyn_fixed_mclachlan"].status == "runnable"
    assert hubbard["dyn_product_formula_envelope"].status == "runnable"
    assert hubbard["dyn_qdrift"].status == "runnable"
    assert hubbard["dyn_fixed_pvqd"].status == "runnable"
    assert hubbard["dyn_adaptive_pvqd"].status == "runnable"
    assert hubbard["dyn_avqds"].status == "runnable"
    assert hubbard["dyn_avqds_t"].status == "runnable"
    assert hubbard["dyn_exact_reference"].command
    assert hubbard["dyn_fixed_mclachlan"].command
    assert hubbard["dyn_product_formula_envelope"].command
    assert hubbard["dyn_qdrift"].command
    assert hubbard["dyn_fixed_pvqd"].command
    assert hubbard["dyn_adaptive_pvqd"].command
    assert hubbard["dyn_avqds"].command
    assert hubbard["dyn_avqds_t"].command
    assert hubbard["dyn_controller_full"].status == "runnable"
    assert hubbard["dyn_controller_full"].command
    assert hubbard["dyn_controller_ablation_matrix"].status == "runnable"
    assert hubbard["dyn_controller_ablation_matrix"].command
    assert hubbard["dyn_exact_reference"].metadata["dispatch"] == "generic_realtime_neutral"
    assert hubbard["dyn_fixed_pvqd"].metadata["dispatch"] == "generic_repo_native_comparator"
    assert hubbard["dyn_adaptive_pvqd"].metadata["dispatch"] == "generic_repo_native_comparator"
    assert hubbard["dyn_avqds"].metadata["dispatch"] == "generic_repo_native_comparator"
    assert hubbard["dyn_avqds_t"].metadata["dispatch"] == "generic_repo_native_comparator"
    assert hubbard["dyn_controller_full"].metadata["dispatch"] == "generic_controller_ablation_row"
    assert hubbard["dyn_controller_ablation_matrix"].metadata["dispatch"] == "generic_controller_ablation_matrix"
    assert hubbard["dyn_exact_reference"].metadata["table_class"] == "fermionic_lattice"
    assert hubbard["dyn_fixed_mclachlan"].metadata["source_table_class"] == "fermionic_lattice"
    assert hubbard["dyn_fixed_mclachlan"].metadata["tuning_granularity"] == "coarse_hamiltonian_class"
    assert hubbard["dyn_fixed_mclachlan"].metadata["tuning_class"] == "fermionic"
    assert hubbard["dyn_fixed_mclachlan"].metadata["static_scaffold_scope"] == "benchmark_point"
    assert hubbard["dyn_fixed_mclachlan"].metadata["class_tuned_result_locked"] is False
    assert "hubbard_dynamics_default" not in hubbard["dyn_fixed_mclachlan"].metadata["settings_id"]
    assert hh["dyn_exact_reference"].metadata["settings_source"] == "hh_legacy_not_class_locked"

    assert hubbard["dyn_vff_like"].status == "skipped_unsupported"
    assert not hubbard["dyn_vff_like"].command


@pytest.mark.parametrize(
    ("algorithm_id", "expected_mean_error", "expected_label"),
    [
        ("dyn_exact_reference", 0.0, "diagnostic exact reference"),
        ("dyn_fixed_mclachlan", (9.0 + 0.4) / 2.0, "fixed-scaffold McLachlan"),
    ],
)
def test_generic_row_dispatch_reuses_neutral_realtime_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    algorithm_id: str,
    expected_mean_error: float,
    expected_label: str,
) -> None:
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _fake_realtime_payload()

    monkeypatch.setattr(rows_mod.realtime, "run_from_args", fake_run_from_args)
    case = get_generic_dynamics_case("hubbard_dynamics_default")

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id=algorithm_id,
        output_dir=tmp_path,
    ).to_dict()

    args = captured["args"]
    if algorithm_id == "dyn_fixed_mclachlan":
        assert args.checkpoint_controller_mode == "observable_v1"
        assert args.checkpoint_controller_reference_mode == "off"
        assert args.diagnostic_exact_reference_mode == "benchmark_exact"
        assert args.checkpoint_controller_noise_mode == "ideal"
    else:
        assert args.checkpoint_controller_mode == "exact_v1"
        assert args.checkpoint_controller_reference_mode == "benchmark_exact"
        assert args.diagnostic_exact_reference_mode == "benchmark_exact"
    assert args.artifact_json == case.artifact_json
    assert bool(args.lock_fixed_manifold) is (algorithm_id == "dyn_fixed_mclachlan")

    assert row["status"] == "completed"
    assert row["qpu_faithful"] is (algorithm_id == "dyn_fixed_mclachlan")
    assert row["exact_assisted"] is (algorithm_id == "dyn_exact_reference")
    assert row["diagnostic"] is True
    assert row["metrics"]["mean_abs_energy_total_error"] == pytest.approx(expected_mean_error)
    assert row["table_fields"]["mean_abs_energy_total_error"] == pytest.approx(expected_mean_error)
    expected_obs_error = 0.0 if algorithm_id == "dyn_exact_reference" else 0.25
    assert row["table_fields"]["epsilon_obs_2"] == pytest.approx(expected_obs_error)
    if algorithm_id == "dyn_exact_reference":
        assert row["metrics"]["epsilon_obs_2_policy"] == "exact_reference_self_comparison"
        assert row["metrics"]["fidelity_policy"] == "exact_reference_self_comparison"
        assert row["table_fields"]["one_minus_min_fidelity_exact"] == pytest.approx(0.0)
    assert row["table_fields"]["compiled_count_2q_total"] == 10
    assert row["table_fields"]["compiled_depth_total"] == 21
    assert row["table_fields"]["table_status_label"] == expected_label
    assert row["provenance"]["tuning_class"] == "fermionic"
    assert row["provenance"]["static_scaffold_scope"] == "benchmark_point"
    assert row["provenance"]["static_scaffold_source"] == case.artifact_json
    assert row["provenance"]["class_tuned_result_locked"] is False
    assert row["provenance"]["tuning_provenance"]["settings_id"] == row["provenance"]["settings_id"]
    assert (tmp_path / "command.json").exists()
    assert (tmp_path / "raw_payload.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "summary.json").exists()


def test_fixed_mclachlan_case_metadata_can_force_rk4(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_from_args(args):
        captured["args"] = args
        return _fake_realtime_payload()

    monkeypatch.setattr(rows_mod.realtime, "run_from_args", fake_run_from_args)
    base_case = get_generic_dynamics_case("hubbard_dynamics_default")
    case = replace(
        base_case,
        metadata={
            **dict(base_case.metadata or {}),
            "fixed_mclachlan_integrator_policy": "rk4",
        },
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_fixed_mclachlan",
        output_dir=tmp_path,
    ).to_dict()

    args = captured["args"]
    assert args.checkpoint_controller_integrator_policy == "rk4"
    payload_keys = row["provenance"]["tuning_provenance"]["settings_payload_keys"]
    assert "integrator_policy" in payload_keys
    assert "integrator_policy_override_source" in payload_keys



def test_skipped_explicit_case_does_not_claim_generic_dispatch(tmp_path: Path) -> None:
    artifact = tmp_path / "molecular_seed.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "cases.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "molecular_dynamics_default",
                        "family": "molecular_restricted_closed_shell",
                        "table_class": "molecular_chemistry",
                        "artifact_json": str(artifact),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    jobs = bench.build_dynamics_jobs(
        output_root=tmp_path / "out",
        families=("molecular_restricted_closed_shell",),
        algorithm_ids=("dyn_exact_reference",),
        include_skipped=True,
        case_manifest=manifest,
    )

    assert len(jobs) == 1
    assert jobs[0].status == "skipped_not_implemented"
    assert not jobs[0].command
    assert jobs[0].metadata["dispatch"] == "generic_case_skipped"


def test_manifest_command_forwards_qiskit_parity_flags(tmp_path: Path) -> None:
    jobs = bench.build_dynamics_jobs(
        output_root=tmp_path / "out",
        families=("hubbard",),
        algorithm_ids=("dyn_product_formula_envelope",),
        include_skipped=True,
        qiskit_dynamics_mode="parity_required",
        qiskit_qubit_cap=3,
    )

    assert len(jobs) == 1
    command = list(jobs[0].command)
    assert "--qiskit-dynamics-mode" in command
    assert command[command.index("--qiskit-dynamics-mode") + 1] == "parity_required"
    assert command[command.index("--qiskit-qubit-cap") + 1] == "3"
    assert jobs[0].metadata["qiskit_dynamics"]["mode"] == "parity_required"
    assert jobs[0].metadata["qiskit_dynamics"]["support_scope"] == "benchmark_local_parity_only"

    uncapped_jobs = bench.build_dynamics_jobs(
        output_root=tmp_path / "out_uncapped",
        families=("hubbard",),
        algorithm_ids=("dyn_product_formula_envelope",),
        include_skipped=True,
        qiskit_dynamics_mode="parity",
        qiskit_qubit_cap=None,
    )
    uncapped_command = list(uncapped_jobs[0].command)
    assert uncapped_command[uncapped_command.index("--qiskit-qubit-cap") + 1] == "none"
    parsed = bench.build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--qiskit-dynamics-mode",
            "parity",
            "--qiskit-qubit-cap",
            "none",
        ]
    )
    assert parsed.qiskit_qubit_cap is None


def test_run_single_calls_isolated_registry_for_non_hh_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_isolated_benchmark(*, case, algorithm_id, output_dir):
        captured["case"] = case
        captured["algorithm_id"] = algorithm_id
        captured["output_dir"] = output_dir
        return DynamicsBenchmarkRow(
            family=case.family,
            table_class=case.table_class,
            case_id=case.case_id,
            algorithm_id=algorithm_id,
            method_label="Exact/Krylov reference dynamics",
            status="completed",
            reason="unit",
            qpu_faithful=False,
            exact_assisted=True,
            diagnostic=True,
            artifact_json="raw_payload.json",
            metrics={"mean_abs_energy_total_error": 0.0},
            resources={},
            provenance={"benchmark_only": True},
            table_fields=DynamicsTableFields(mean_abs_energy_total_error=0.0),
        )

    monkeypatch.setattr(benchmark_registry, "run_isolated_benchmark", fake_run_isolated_benchmark)

    payload = bench.run_single(
        family="hubbard",
        case_id="hubbard_dynamics_default",
        algorithm_id="dyn_exact_reference",
        output_dir=tmp_path,
    )

    assert captured["case"].family == "hubbard"
    assert captured["case"].case_id == "hubbard_dynamics_default"
    assert captured["algorithm_id"] == "dyn_exact_reference"
    assert captured["output_dir"] == tmp_path
    assert payload["status"] == "completed"
    assert payload["qpu_faithful"] is False
    assert payload["exact_assisted"] is True


def test_run_single_forwards_qiskit_parity_metadata_to_generic_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_isolated_benchmark(*, case, algorithm_id, output_dir):
        captured["case"] = case
        return DynamicsBenchmarkRow(
            family=case.family,
            table_class=case.table_class,
            case_id=case.case_id,
            algorithm_id=algorithm_id,
            method_label="Product Formula",
            status="completed",
            reason="unit",
            qpu_faithful=True,
            exact_assisted=False,
            diagnostic=True,
            artifact_json="raw_payload.json",
            metrics={"mean_abs_energy_total_error": 0.0},
            resources={},
            provenance={"benchmark_only": True},
            table_fields=DynamicsTableFields(mean_abs_energy_total_error=0.0),
        )

    monkeypatch.setattr(benchmark_registry, "run_isolated_benchmark", fake_run_isolated_benchmark)

    payload = bench.run_single(
        family="hubbard",
        case_id="hubbard_dynamics_default",
        algorithm_id="dyn_product_formula_envelope",
        output_dir=tmp_path,
        qiskit_dynamics_mode="parity",
        qiskit_qubit_cap=2,
    )

    assert payload["status"] == "completed"
    assert captured["case"].metadata["qiskit_dynamics"]["mode"] == "parity"
    assert captured["case"].metadata["qiskit_dynamics"]["qubit_cap"] == 2
    assert captured["case"].metadata["qiskit_dynamics"]["time_segmentation"] == "match_native_interval"


def test_run_single_calls_isolated_registry_for_non_hh_ablation_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_isolated_benchmark(*, case, algorithm_id, output_dir):
        captured["case"] = case
        captured["algorithm_id"] = algorithm_id
        captured["output_dir"] = output_dir
        return {
            "schema": "generic_controller_ablation_matrix_v1",
            "algorithm_id": "dyn_controller_ablation_matrix",
            "status_counts": {"completed": 1},
            "rows": [],
        }

    monkeypatch.setattr(benchmark_registry, "run_isolated_benchmark", fake_run_isolated_benchmark)

    payload = bench.run_single(
        family="hubbard",
        case_id="hubbard_dynamics_default",
        algorithm_id="dyn_controller_ablation_matrix",
        output_dir=tmp_path,
    )

    assert captured["case"].family == "hubbard"
    assert captured["case"].case_id == "hubbard_dynamics_default"
    assert captured["algorithm_id"] == "dyn_controller_ablation_matrix"
    assert captured["output_dir"] == tmp_path
    assert payload["schema"] == "generic_controller_ablation_matrix_v1"
    assert payload["algorithm_id"] == "dyn_controller_ablation_matrix"


def test_run_single_calls_isolated_registry_for_non_hh_full_controller_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_run_isolated_benchmark(*, case, algorithm_id, output_dir):
        captured["case"] = case
        captured["algorithm_id"] = algorithm_id
        captured["output_dir"] = output_dir
        return {
            "schema": "generic_dynamics_benchmark_row_v1",
            "algorithm_id": "dyn_controller_full",
            "status": "completed",
        }

    monkeypatch.setattr(benchmark_registry, "run_isolated_benchmark", fake_run_isolated_benchmark)

    payload = bench.run_single(
        family="hubbard",
        case_id="hubbard_dynamics_default",
        algorithm_id="dyn_controller_full",
        output_dir=tmp_path,
    )

    assert captured["case"].family == "hubbard"
    assert captured["case"].case_id == "hubbard_dynamics_default"
    assert captured["algorithm_id"] == "dyn_controller_full"
    assert captured["output_dir"] == tmp_path
    assert payload["algorithm_id"] == "dyn_controller_full"
    assert payload["status"] == "completed"


def test_run_single_non_hh_unwired_comparator_writes_structured_skip(tmp_path: Path) -> None:
    payload = bench.run_single(
        family="hubbard",
        case_id="hubbard_dynamics_default",
        algorithm_id="dyn_vff_like",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_unsupported"
    assert payload["reason"]
    assert (tmp_path / "skip.json").exists()
    assert (tmp_path / "rows.json").exists()
    skip = json.loads((tmp_path / "skip.json").read_text(encoding="utf-8"))
    assert skip["schema"] == "generic_dynamics_benchmark_skip_v1"
    assert skip["status"] == "skipped_unsupported"

def _single_qubit_poly(label: str, coeff: float) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(1, ps=label, pc=coeff))
    poly._reduce()
    return poly


def _single_qubit_term(label: str, pauli: str, coeff: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(label=label, polynomial=_single_qubit_poly(pauli, coeff))


def _single_qubit_runtime_input(
    *,
    candidate_pool_complete: bool = True,
    selected_pauli: str = "x",
    candidate_pauli: str = "y",
    include_grouped_candidate: bool = False,
) -> SimpleNamespace:
    h_poly = _single_qubit_poly("x", 0.7)
    selected = (_single_qubit_term(f"{selected_pauli}_rotation", selected_pauli, 1.0),)
    candidate = _single_qubit_term(f"{candidate_pauli}_rotation", candidate_pauli, 1.0)
    extra_candidates = []
    if include_grouped_candidate:
        grouped = _single_qubit_term("grouped_exact_rotation", "z", 1.0)
        grouped.execution_mode = "grouped_exact"
        extra_candidates.append(grouped)
    layout = build_parameter_layout(selected)
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    return SimpleNamespace(
        h_poly=h_poly,
        psi_ref=psi_ref,
        psi_initial=psi_ref.copy(),
        selected_terms=selected,
        candidate_pool_terms=selected + tuple(extra_candidates) + (candidate,),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool" if candidate_pool_complete else "selected_terms_only",
            pool_key="unit_pool" if candidate_pool_complete else None,
            completeness="complete" if candidate_pool_complete else "selected_only",
        ),
        base_layout=layout,
        theta_runtime=np.zeros(layout.runtime_parameter_count, dtype=float),
        structure_locked=False,
        provenance={"artifact_json": "unit_seed.json"},
        resolved_problem=SimpleNamespace(
            family_key="hubbard",
            hamiltonian=h_poly,
            request=SimpleNamespace(num_sites=1),
        ),
    )


def _native_comparator_case(tmp_path: Path, *, metadata: dict | None = None) -> DynamicsBenchmarkCase:
    artifact = tmp_path / "unit_seed.json"
    artifact.write_text("{}", encoding="utf-8")
    return DynamicsBenchmarkCase(
        case_id="unit_hubbard",
        family="hubbard",
        table_class="fermionic_lattice",
        artifact_json=str(artifact),
        t_final=1.0,
        num_times=3,
        metadata={} if metadata is None else metadata,
    )


def test_fixed_mclachlan_correctness_allows_terminal_no_integrator_sample(tmp_path: Path) -> None:
    payload = _fake_realtime_payload()
    payload["trajectory"][-1]["integrator_used"] = "none"
    payload["trajectory"][-1]["theta_update_l2"] = 0.0
    payload["ledger"] = list(payload["trajectory"])
    payload["provenance"] = {"exact_reference_controller_inputs": False}

    correctness = rows_mod.build_fixed_mclachlan_correctness_sidecar(
        case=_native_comparator_case(tmp_path),
        payload=payload,
    )

    assert correctness["passed"] is True
    integrator_check = next(
        check for check in correctness["checks"] if check["check_id"] == "integrator_policy_and_step_semantics"
    )
    assert integrator_check["passed"] is True
    assert integrator_check["details"]["integrator_used_values"] == ["none", "rk4"]


def test_fixed_mclachlan_correctness_rejects_nonterminal_no_integrator_sample(tmp_path: Path) -> None:
    payload = _fake_realtime_payload()
    payload["trajectory"][0]["integrator_used"] = "none"
    payload["provenance"] = {"exact_reference_controller_inputs": False}

    correctness = rows_mod.build_fixed_mclachlan_correctness_sidecar(
        case=_native_comparator_case(tmp_path),
        payload=payload,
    )

    assert correctness["passed"] is False
    integrator_check = next(
        check for check in correctness["checks"] if check["check_id"] == "integrator_policy_and_step_semantics"
    )
    assert integrator_check["passed"] is False
    assert integrator_check["details"]["bad_rows"] == [
        {"checkpoint_index": 0, "reason": "bad_used", "value": "none"}
    ]


def test_fixed_mclachlan_motion_helper_compares_same_observable_over_time() -> None:
    rows = [
        {
            "energy_total_exact": 1.0,
            "primary_density_exact": 0.0,
            "site_occupations_exact": [1.0, 0.0],
        },
        {
            "energy_total_exact": 1.0,
            "primary_density_exact": 0.0,
            "site_occupations_exact": [1.0, 0.0],
        },
    ]

    assert rows_mod._movement_from_rows(
        rows,
        ("energy_total_exact", "primary_density_exact", "site_occupations_exact"),
    ) == pytest.approx(0.0)

    rows[1]["site_occupations_exact"] = [0.9, 0.1]
    assert rows_mod._movement_from_rows(
        rows,
        ("energy_total_exact", "primary_density_exact", "site_occupations_exact"),
    ) == pytest.approx(0.1)


def test_avqds_candidate_helpers_respect_zero_limit_and_transfer_theta_by_label() -> None:
    x_term = _single_qubit_term("x_rotation", "x", 1.0)
    y_term = _single_qubit_term("y_rotation", "y", 1.0)
    grouped_term = _single_qubit_term("grouped_rotation", "z", 1.0)
    grouped_term.execution_mode = "grouped_exact"

    assert rows_mod._candidate_indices_for_avqds(
        candidate_pool=(x_term, y_term),
        used_labels=set(),
        candidate_limit=0,
    ) == []
    assert rows_mod._candidate_indices_for_avqds(
        candidate_pool=(x_term, grouped_term, y_term),
        used_labels=set(),
        candidate_limit=None,
    ) == [0, 2]

    old_layout = build_parameter_layout((x_term,))
    reordered_layout = build_parameter_layout((y_term, x_term))
    copied = rows_mod._copy_theta_by_layout_blocks(
        old_theta=np.asarray([0.37], dtype=float),
        old_layout=old_layout,
        new_layout=reordered_layout,
    )

    assert copied[0] == pytest.approx(0.0)
    assert copied[1] == pytest.approx(0.37)


def test_native_comparator_loader_forwards_case_append_pool_family_when_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_loader(
        artifact_json,
        *,
        loader_mode,
        tag,
        generator_family,
        fallback_family,
        append_pool_family,
    ):
        captured.update(
            {
                "artifact_json": artifact_json,
                "loader_mode": loader_mode,
                "tag": tag,
                "generator_family": generator_family,
                "fallback_family": fallback_family,
                "append_pool_family": append_pool_family,
            }
        )
        return _single_qubit_runtime_input()

    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", fake_loader)
    case = DynamicsBenchmarkCase(
        case_id="unit_hubbard",
        family="hubbard",
        table_class="fermionic_lattice",
        artifact_json=str(tmp_path / "unit_seed.json"),
        append_pool_family="unit_append_pool",
        t_final=1.0,
        num_times=3,
    )
    Path(case.artifact_json).write_text("{}", encoding="utf-8")

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_product_formula_envelope",
        output_dir=tmp_path / "forward_append_pool",
    ).to_dict()

    assert row["status"] == "completed"
    assert captured["append_pool_family"] == "unit_append_pool"
    assert captured["loader_mode"] == case.loader_mode
    assert captured["generator_family"] == case.generator_family
    assert captured["fallback_family"] == case.fallback_family


def _require_qiskit_for_generic_parity() -> None:
    from pipelines.exact_bench.qiskit_pauli_tools import has_qiskit_pauli_support

    if not has_qiskit_pauli_support():
        pytest.skip("optional Qiskit is not installed")


def test_generic_product_formula_parity_mode_writes_qiskit_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _require_qiskit_for_generic_parity()
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    case = _native_comparator_case(
        tmp_path,
        metadata={"qiskit_dynamics": {"mode": "parity_required", "qubit_cap": 4}},
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_product_formula_envelope",
        output_dir=tmp_path / "pf_qiskit",
    ).to_dict()

    assert row["status"] == "completed"
    assert row["metrics"]["qiskit_parity_status"] == "ok"
    assert row["metrics"]["qiskit_parity_passed"] is True
    assert row["provenance"]["qiskit_boundary"] == "pipelines.exact_bench_only"
    sidecar = json.loads((tmp_path / "pf_qiskit" / "qiskit_parity.json").read_text(encoding="utf-8"))
    assert sidecar["support_scope"] == "product_formula_sequence_parity_only"
    assert sidecar["passed"] is True


def test_generic_qdrift_and_pvqd_parity_modes_are_additive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _require_qiskit_for_generic_parity()
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    metadata = {
        "qiskit_dynamics": {"mode": "parity_required", "qubit_cap": 4},
        "qdrift_samples_per_interval": 2,
        "qdrift_rng_seed": 3,
        "pvqd_optimizer_maxiter": 4,
    }
    for algorithm_id, support_scope in (
        ("dyn_qdrift", "qdrift_realized_sample_plan_parity_only"),
        ("dyn_fixed_pvqd", "fixed_pvqd_scaffold_target_projection_component_parity"),
    ):
        case = _native_comparator_case(tmp_path, metadata=metadata)
        output_dir = tmp_path / f"{algorithm_id}_qiskit"
        row = rows_mod.run_generic_dynamics_row(
            case=case,
            algorithm_id=algorithm_id,
            output_dir=output_dir,
        ).to_dict()
        assert row["status"] == "completed"
        assert row["metrics"]["qiskit_parity_status"] == "ok"
        assert row["metrics"]["qiskit_parity_passed"] is True
        sidecar = json.loads((output_dir / "qiskit_parity.json").read_text(encoding="utf-8"))
        assert sidecar["support_scope"] == support_scope
        assert sidecar["passed"] is True


def test_generic_fixed_mclachlan_required_parity_uses_post_run_scaffold_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _require_qiskit_for_generic_parity()
    captured = {}
    layout = build_parameter_layout((_single_qubit_term("x_rotation", "x", 1.0),))
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    payload = _fake_realtime_payload()
    payload["fixed_scaffold_parity_payload"] = {
        "schema": "fixed_scaffold_qiskit_post_run_payload_v1",
        "psi_ref": _complex_payload(psi_ref),
        "fixed_layout": serialize_layout(layout),
        "layout_stable": True,
        "checkpoint_count": 1,
        "checkpoints": [
            {
                "checkpoint_index": 0,
                "time": 0.0,
                "theta_runtime": [0.0],
                "native_state": _complex_payload(psi_ref),
                "energy_total_controller": 0.0,
                "hamiltonian_terms_exyz": [
                    {"pauli_exyz": "x", "coeff_re": 0.7, "coeff_im": 0.0}
                ],
            }
        ],
    }

    def fake_run_from_args(args):
        captured["args"] = args
        return payload

    monkeypatch.setattr(rows_mod.realtime, "run_from_args", fake_run_from_args)
    case = _native_comparator_case(
        tmp_path,
        metadata={"qiskit_dynamics": {"mode": "parity_required", "qubit_cap": 4}},
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_fixed_mclachlan",
        output_dir=tmp_path / "fixed_mcl_qiskit",
    ).to_dict()

    assert captured["args"].emit_fixed_scaffold_qiskit_parity_payload is True
    assert row["status"] == "completed"
    assert row["metrics"]["qiskit_parity_status"] == "ok"
    assert row["metrics"]["qiskit_parity_passed"] is True
    assert row["metrics"]["qiskit_max_state_l2"] <= 1.0e-12
    assert row["metrics"]["mclachlan_correctness_passed"] is True
    assert row["provenance"]["qiskit_boundary"] == "pipelines.exact_bench_only"
    assert row["provenance"]["correctness_sidecar_name"] == "mclachlan_correctness.json"
    sidecar = json.loads(
        (tmp_path / "fixed_mcl_qiskit" / "qiskit_parity.json").read_text(encoding="utf-8")
    )
    assert sidecar["support_scope"] == "fixed_mclachlan_post_run_scaffold_state_and_observable_parity"
    assert sidecar["passed"] is True
    assert sidecar["qiskit_used_in_online_controller"] is False
    correctness = json.loads(
        (tmp_path / "fixed_mcl_qiskit" / "mclachlan_correctness.json").read_text(encoding="utf-8")
    )
    assert correctness["schema"] == "fixed_mclachlan_correctness_v1"
    assert correctness["support_scope"] == "fixed_mclachlan_metric_rhs_solve_integrator_and_nonfrozen_correctness"
    assert correctness["passed"] is True
    assert correctness["qiskit_scaffold_parity_is_separate"] is True



def test_fixed_mclachlan_correctness_failure_blocks_scaffold_parity_only_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _fake_realtime_payload()
    for row in payload["trajectory"]:
        row.pop("baseline_geometry", None)

    monkeypatch.setattr(rows_mod.realtime, "run_from_args", lambda args: payload)
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_fixed_mclachlan",
        output_dir=tmp_path / "fixed_mcl_correctness_fail",
    ).to_dict()

    assert row["status"] == "failed"
    assert "mclachlan_correctness.json required but did not pass" in row["reason"]
    correctness = json.loads(
        (tmp_path / "fixed_mcl_correctness_fail" / "mclachlan_correctness.json").read_text(encoding="utf-8")
    )
    assert correctness["passed"] is False
    assert any(check["check_id"] == "metric_force_rhs_geometry_fields" and check["passed"] is False for check in correctness["checks"])


def test_fixed_mclachlan_missing_fixed_manifold_telemetry_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _fake_realtime_payload()
    for row in payload["trajectory"]:
        row.pop("action_kind", None)
        row.pop("runtime_parameter_count_delta", None)

    monkeypatch.setattr(rows_mod.realtime, "run_from_args", lambda args: payload)
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_fixed_mclachlan",
        output_dir=tmp_path / "fixed_mcl_missing_fixed_telemetry",
    ).to_dict()

    assert row["status"] == "failed"
    correctness = json.loads(
        (tmp_path / "fixed_mcl_missing_fixed_telemetry" / "mclachlan_correctness.json").read_text(
            encoding="utf-8"
        )
    )
    fixed_check = next(
        check for check in correctness["checks"] if check["check_id"] == "fixed_manifold_no_append_prune_semantics"
    )
    assert fixed_check["passed"] is False
    reasons = {bad["reason"] for bad in fixed_check["details"]["bad_rows"]}
    assert {"missing_action_kind", "missing_runtime_parameter_count_delta"}.issubset(reasons)


def test_fixed_mclachlan_exact_decision_leakage_in_provenance_is_preserved_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _fake_realtime_payload()
    payload["provenance"] = {
        "uses_reference_for_decision": True,
        "exact_reference_controller_inputs": True,
    }

    monkeypatch.setattr(rows_mod.realtime, "run_from_args", lambda args: payload)
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_fixed_mclachlan",
        output_dir=tmp_path / "fixed_mcl_leakage",
    ).to_dict()

    assert row["status"] == "failed"
    assert row["provenance"]["uses_reference_for_decision"] is True
    assert row["provenance"]["exact_reference_controller_inputs"] is True
    correctness = json.loads(
        (tmp_path / "fixed_mcl_leakage" / "mclachlan_correctness.json").read_text(encoding="utf-8")
    )
    exact_check = next(check for check in correctness["checks"] if check["check_id"] == "exact_reference_diagnostic_only")
    assert exact_check["passed"] is False
    assert {entry["field"] for entry in exact_check["details"]["truthy_exact_decision_flags"]} >= {
        "uses_reference_for_decision",
        "exact_reference_controller_inputs",
    }


def test_parity_correctness_required_flag_blocks_qiskit_supported_comparator_without_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    case = _native_comparator_case(
        tmp_path,
        metadata={"require_parity_correctness_sidecars": True},
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_product_formula_envelope",
        output_dir=tmp_path / "pf_missing_required_qiskit",
    ).to_dict()

    assert row["status"] == "failed"
    assert row["reason"] == "qiskit parity required but no parity sidecar was produced"
    assert not (tmp_path / "pf_missing_required_qiskit" / "qiskit_parity.json").exists()



def test_fixed_scaffold_qiskit_parity_observer_serializes_layout_theta_and_terms() -> None:
    layout = build_parameter_layout((_single_qubit_term("x_rotation", "x", 1.0),))
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    runtime_input = SimpleNamespace(psi_ref=psi_ref)
    observer = rows_mod.realtime.FixedScaffoldQiskitParityObserver(runtime_input=runtime_input)
    step_hamiltonian = SimpleNamespace(h_poly=_single_qubit_poly("x", 0.7))

    observer.on_checkpoint(
        {
            "checkpoint_index": 0,
            "time": 0.0,
            "time_stop": 0.5,
            "physical_time": 0.0,
            "layout_at_checkpoint": layout,
            "theta_runtime_at_checkpoint": np.asarray([0.25], dtype=float),
            "psi_current": psi_ref,
            "energy_total_controller": 0.0,
            "controller_obs": {"primary_density": 1.0},
            "trajectory_row": {"energy_total": 0.0},
            "step_hamiltonian": step_hamiltonian,
        }
    )
    payload = observer.to_payload()

    assert payload["schema"] == "fixed_scaffold_qiskit_post_run_payload_v1"
    assert payload["qiskit_used_in_online_controller"] is False
    assert payload["fixed_layout"]["runtime_parameter_count"] == 1
    assert payload["checkpoints"][0]["theta_runtime"] == [0.25]
    assert payload["checkpoints"][0]["hamiltonian_terms_exyz"] == [
        {"pauli_exyz": "x", "coeff_re": 0.7, "coeff_im": 0.0}
    ]



def test_generic_product_formula_dispatch_emits_stable_schema_and_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_product_formula_envelope",
        output_dir=tmp_path / "pf",
    ).to_dict()

    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_product_formula_envelope"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["method_kind"] == "product_formula_envelope"
    assert row["metrics"]["candidate_orders"] == [1, 2]
    assert row["metrics"]["selected_order"] == 2
    assert row["metrics"]["selection_uses_exact_reference"] is False
    assert row["table_fields"]["table_status_label"] == "product-formula/Suzuki envelope"
    assert row["table_fields"]["mean_abs_energy_total_error"] == pytest.approx(0.0, abs=1e-12)
    assert row["table_fields"]["compiled_count_2q_total"] == 0
    assert row["table_fields"]["compiled_depth_total"] > 0
    assert row["resources"]["resource_policy"] == "repo_native_pauli_rotation_proxy_no_qiskit"
    assert row["resources"]["state_at_time_scope"] == "one_product_formula_interval"
    assert row["provenance"]["route_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert row["provenance"]["benchmark_only"] is True
    assert row["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert row["provenance"]["controller_decisions_modified"] is False
    assert row["provenance"]["exact_reference_controller_inputs"] is False
    assert (tmp_path / "pf" / "raw_payload.json").exists()
    raw = json.loads((tmp_path / "pf" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "generic_product_formula_envelope_benchmark_v1"
    assert raw["parameter_manifest"]["time_dependence"] == "static_hamiltonian_only"
    assert raw["parameter_manifest"]["drive_included"] is False
    assert raw["provenance"]["benchmark_only"] is True
    assert raw["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert raw["provenance"]["controller_decisions_modified"] is False


def test_generic_qdrift_dispatch_emits_stable_schema_status_and_resource_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    case = _native_comparator_case(
        tmp_path,
        metadata={"qdrift_samples_per_interval": 3, "qdrift_rng_seed": 11},
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_qdrift",
        output_dir=tmp_path / "qdrift",
    ).to_dict()

    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_qdrift"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["method_kind"] == "randomized_product_formula"
    assert row["metrics"]["randomization_family"] == "qdrift"
    assert row["metrics"]["samples_per_interval"] == 3
    assert row["metrics"]["rng_seed"] == 11
    assert row["metrics"]["sampled_rotation_count"] == 6
    assert row["table_fields"]["table_status_label"] == "qDRIFT randomized product formula"
    assert row["table_fields"]["compiled_count_2q_total"] == 0
    assert row["table_fields"]["compiled_depth_total"] == 6
    assert row["resources"]["resource_policy"] == "repo_native_pauli_rotation_proxy_no_qiskit"
    assert row["resources"]["state_at_time_scope"] == "representative_interval0_qdrift_sample"
    assert row["resources"]["full_horizon_scope"] == "all_sampled_qdrift_microsteps"
    assert row["provenance"]["route_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert row["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    raw = json.loads((tmp_path / "qdrift" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "generic_qdrift_benchmark_v1"
    assert len(raw["qdrift_intervals"]) == 2
    assert raw["provenance"]["qdrift_sampling_depends_on_exact_fields"] is False


def test_generic_fixed_pvqd_dispatch_emits_qpu_faithful_product_formula_target_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    case = _native_comparator_case(
        tmp_path,
        metadata={"pvqd_optimizer_maxiter": 6, "pvqd_overlap_tol": 1.0e-6},
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_fixed_pvqd",
        output_dir=tmp_path / "pvqd",
    ).to_dict()

    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_fixed_pvqd"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["method_kind"] == "fixed_pvqd"
    assert row["metrics"]["pvqd_step_count"] == 2
    assert row["metrics"]["pvqd_nfev_total"] > 0
    assert row["metrics"]["pvqd_target_depends_on_exact_interval_propagation"] is False
    assert row["metrics"]["pvqd_target_policy"] == "product_formula_circuit_step"
    assert row["metrics"]["exact_fields_reporting_only"] is True
    assert row["table_fields"]["table_status_label"] == "fixed pVQD product-formula target"
    assert row["table_fields"]["mean_abs_energy_total_error"] is not None
    assert row["resources"]["resource_policy"] == "repo_native_pauli_rotation_proxy_no_qiskit"
    assert row["resources"]["state_at_time_scope"] == "generic_fixed_pvqd_state_scaffold"
    raw = json.loads((tmp_path / "pvqd" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "generic_fixed_pvqd_benchmark_v1"
    assert raw["row_contract"] == {"qpu_faithful": True, "exact_assisted": False, "diagnostic": True}
    assert raw["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert raw["provenance"]["controller_decisions_modified"] is False
    assert raw["provenance"]["exact_interval_targets_used_by_comparator"] is False


def test_generic_adaptive_pvqd_dispatch_emits_qpu_faithful_product_formula_target_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _single_qubit_runtime_input(selected_pauli="y", candidate_pauli="x"),
    )
    case = _native_comparator_case(
        tmp_path,
        metadata={
            "adaptive_pvqd_optimizer_maxiter": 8,
            "adaptive_pvqd_overlap_tol": 1.0e-8,
            "adaptive_pvqd_append_loss_threshold": 0.0,
            "adaptive_pvqd_append_min_loss_improvement": 0.0,
            "adaptive_pvqd_append_candidate_limit": 1,
        },
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_adaptive_pvqd",
        output_dir=tmp_path / "adaptive_pvqd",
    ).to_dict()

    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_adaptive_pvqd"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["method_kind"] == "adaptive_pvqd"
    assert row["metrics"]["candidate_pool_complete"] is True
    assert row["metrics"]["candidate_pool_size"] == 2
    assert row["metrics"]["adaptive_pvqd_step_count"] == 2
    assert row["metrics"]["append_candidate_evaluations_total"] >= 1
    assert row["metrics"]["pvqd_target_depends_on_exact_interval_propagation"] is False
    assert row["metrics"]["append_scoring_uses_exact_reference"] is False
    assert row["table_fields"]["table_status_label"] == "adaptive pVQD product-formula target"
    assert row["resources"]["resource_policy"] == "repo_native_pauli_rotation_proxy_no_qiskit"
    assert row["resources"]["state_at_time_scope"] == "generic_adaptive_pvqd_state_scaffold"
    assert row["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert row["provenance"]["exact_data_policy"] == "diagnostic_exact_reference_reporting_only_not_pvqd_target"
    assert row["provenance"]["controller_decisions_modified"] is False
    assert row["provenance"]["exact_reference_controller_inputs"] is False
    raw = json.loads((tmp_path / "adaptive_pvqd" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "generic_adaptive_pvqd_benchmark_v1"
    assert raw["row_contract"] == {"qpu_faithful": True, "exact_assisted": False, "diagnostic": True}
    assert raw["pvqd_steps"][0]["append_candidate_evaluations"] <= 1
    assert raw["provenance"]["exact_data_policy"] == "diagnostic_exact_reference_reporting_only_not_pvqd_target"
    assert raw["provenance"]["append_scoring_uses_exact_reference"] is False


def test_generic_adaptive_pvqd_incomplete_candidate_pool_writes_structured_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _single_qubit_runtime_input(candidate_pool_complete=False),
    )
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_adaptive_pvqd",
        output_dir=tmp_path / "adaptive_pvqd_skip",
    ).to_dict()

    assert row["status"] == "skipped_unsupported"
    assert "adaptive pVQD comparator requires complete candidate pool" in row["reason"]
    assert row["qpu_faithful"] is False
    assert row["exact_assisted"] is False
    assert (tmp_path / "adaptive_pvqd_skip" / "skip.json").exists()
    skip = json.loads((tmp_path / "adaptive_pvqd_skip" / "skip.json").read_text(encoding="utf-8"))
    assert skip["schema"] == "generic_dynamics_benchmark_skip_v1"
    assert skip["status"] == "skipped_unsupported"


def test_generic_avqds_t_dispatch_emits_qpu_faithful_target_tangent_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _single_qubit_runtime_input(selected_pauli="y", candidate_pauli="x"),
    )
    case = _native_comparator_case(
        tmp_path,
        metadata={
            "avqds_t_append_candidate_limit": 1,
            "avqds_t_append_target_tangent_residual_ratio_threshold": 0.0,
            "avqds_t_append_min_residual_ratio_gain": 0.0,
        },
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_avqds_t",
        output_dir=tmp_path / "avqds_t",
    ).to_dict()

    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_avqds_t"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["method_kind"] == "avqds_t"
    assert row["metrics"]["candidate_pool_complete"] is True
    assert row["metrics"]["candidate_pool_size"] == 2
    assert row["metrics"]["avqds_t_linear_solve_total"] == 2
    assert row["metrics"]["append_candidate_evaluations_total"] >= 1
    assert row["metrics"]["exact_tangent_target_depends_on_exact_interval_propagation"] is False
    assert row["metrics"]["append_scoring_uses_exact_reference"] is False
    assert row["table_fields"]["table_status_label"] == "PF-target adaptive tangent diagnostic"
    assert row["resources"]["resource_policy"] == "repo_native_pauli_rotation_proxy_no_qiskit"
    assert row["resources"]["state_at_time_scope"] == "generic_avqds_t_state_scaffold"
    assert row["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert row["provenance"]["exact_data_policy"] == "diagnostic_exact_reference_reporting_only_not_target_tangent"
    assert row["provenance"]["controller_decisions_modified"] is False
    assert row["provenance"]["exact_reference_controller_inputs"] is False
    assert row["metrics"]["avqds_t_correctness_passed"] is True
    assert row["provenance"]["correctness_sidecar_name"] == "avqds_t_correctness.json"
    raw = json.loads((tmp_path / "avqds_t" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "generic_avqds_t_benchmark_v1"
    assert raw["row_contract"] == {"qpu_faithful": True, "exact_assisted": False, "diagnostic": True}
    assert raw["avqds_t_steps"][0]["append_candidate_evaluations"] <= 1
    assert raw["avqds_t_correctness"]["schema"] == "avqds_t_correctness_v1"
    assert raw["avqds_t_correctness"]["passed"] is True
    assert any(check["check_id"] == "target_tangent_dense_reference_solve" for check in raw["avqds_t_correctness"]["checks"])
    sidecar = json.loads((tmp_path / "avqds_t" / "avqds_t_correctness.json").read_text(encoding="utf-8"))
    assert sidecar == raw["avqds_t_correctness"]
    summary = json.loads((tmp_path / "avqds_t" / "summary.json").read_text(encoding="utf-8"))
    assert summary["paths"]["avqds_t_correctness_json"].endswith("avqds_t_correctness.json")
    assert raw["provenance"]["exact_data_policy"] == "diagnostic_exact_reference_reporting_only_not_target_tangent"
    assert raw["provenance"]["exact_tangent_targets_used_by_comparator"] is False
    assert raw["provenance"]["append_scoring_uses_exact_reference"] is False


def test_generic_avqds_t_incomplete_candidate_pool_writes_structured_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _single_qubit_runtime_input(candidate_pool_complete=False),
    )
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_avqds_t",
        output_dir=tmp_path / "avqds_t_skip",
    ).to_dict()

    assert row["status"] == "skipped_unsupported"
    assert "AVQDS-T comparator requires complete candidate pool" in row["reason"]
    assert row["qpu_faithful"] is False
    assert row["exact_assisted"] is False
    assert (tmp_path / "avqds_t_skip" / "skip.json").exists()
    skip = json.loads((tmp_path / "avqds_t_skip" / "skip.json").read_text(encoding="utf-8"))
    assert skip["schema"] == "generic_dynamics_benchmark_skip_v1"
    assert skip["status"] == "skipped_unsupported"


def test_generic_avqds_dispatch_emits_tangent_metadata_without_exact_assisted_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _single_qubit_runtime_input(include_grouped_candidate=True),
    )
    case = _native_comparator_case(
        tmp_path,
        metadata={
            "avqds_append_candidate_limit": 0,
            "avqds_append_rhs_residual_ratio_threshold": 999.0,
        },
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_avqds",
        output_dir=tmp_path / "avqds",
    ).to_dict()

    assert row["schema"] == "dynamics_benchmark_row_v1"
    assert row["status"] == "completed"
    assert row["algorithm_id"] == "dyn_avqds"
    assert row["qpu_faithful"] is True
    assert row["exact_assisted"] is False
    assert row["diagnostic"] is True
    assert row["metrics"]["method_kind"] == "avqds"
    assert row["metrics"]["candidate_pool_complete"] is True
    assert row["metrics"]["append_candidate_evaluations_total"] == 0
    assert row["metrics"]["exact_fields_reporting_only"] is True
    assert row["metrics"]["avqds_linear_solve_total"] == 2
    assert row["metrics"]["avqds_correctness_passed"] is True
    assert row["provenance"]["correctness_sidecar_name"] == "avqds_correctness.json"
    assert row["table_fields"]["table_status_label"] == "AVQDS tangent diagnostic"
    assert row["resources"]["resource_policy"] == "repo_native_pauli_rotation_proxy_no_qiskit"
    assert row["resources"]["state_at_time_scope"] == "generic_avqds_state_scaffold"
    raw = json.loads((tmp_path / "avqds" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == "generic_avqds_benchmark_v1"
    assert raw["row_contract"]["exact_assisted"] is False
    assert raw["avqds_correctness"]["schema"] == "avqds_correctness_v1"
    assert raw["avqds_correctness"]["passed"] is True
    grouped_check = next(check for check in raw["avqds_correctness"]["checks"] if check["check_id"] == "grouped_exact_candidate_exclusion")
    assert grouped_check["details"]["grouped_exact_candidate_excluded_count"] == 1
    sidecar = json.loads((tmp_path / "avqds" / "avqds_correctness.json").read_text(encoding="utf-8"))
    assert sidecar == raw["avqds_correctness"]
    summary = json.loads((tmp_path / "avqds" / "summary.json").read_text(encoding="utf-8"))
    assert summary["paths"]["avqds_correctness_json"].endswith("avqds_correctness.json")
    assert raw["provenance"]["runner_module"] == "pipelines.time_dynamics.benchmarks.legacy_native"
    assert raw["provenance"]["append_scoring_uses_exact_reference"] is False
    assert raw["provenance"]["exact_fields_reporting_only"] is True


def test_generic_avqds_incomplete_candidate_pool_writes_structured_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rows_mod,
        "load_scaffold_runtime_input",
        lambda *args, **kwargs: _single_qubit_runtime_input(candidate_pool_complete=False),
    )
    case = _native_comparator_case(tmp_path)

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_avqds",
        output_dir=tmp_path / "avqds_skip",
    ).to_dict()

    assert row["status"] == "skipped_unsupported"
    assert "requires complete candidate pool" in row["reason"]
    assert row["qpu_faithful"] is False
    assert row["exact_assisted"] is False
    assert (tmp_path / "avqds_skip" / "skip.json").exists()
    skip = json.loads((tmp_path / "avqds_skip" / "skip.json").read_text(encoding="utf-8"))
    assert skip["schema"] == "generic_dynamics_benchmark_skip_v1"
    assert skip["status"] == "skipped_unsupported"


def test_run_single_missing_non_hh_case_writes_structured_skip(tmp_path: Path) -> None:
    payload = bench.run_single(
        family="hubbard",
        case_id="missing_case",
        algorithm_id="dyn_qdrift",
        output_dir=tmp_path / "missing",
    )

    assert payload["status"] == "skipped_no_runner"
    assert "no explicit generic dynamics case" in payload["reason"]
    assert (tmp_path / "missing" / "skip.json").exists()
    assert (tmp_path / "missing" / "rows.json").exists()


def test_native_comparator_consumes_candidate_class_settings_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: _single_qubit_runtime_input())
    settings_manifest = tmp_path / "class_settings.json"
    settings_manifest.write_text(
        json.dumps(
            {
                "schema": "dynamics_class_settings_lock_manifest_v1",
                "lock_status": "candidate_not_promoted",
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_qdrift",
                        "settings_kind": "comparator",
                        "settings_source": "unit_all_algorithm_candidate",
                        "settings_payload": {
                            "qdrift_samples_per_interval": 3,
                            "qdrift_rng_seed": 123,
                        },
                        "class_tuned_result_locked": False,
                        "candidate_only_not_promoted": True,
                        "promotion_status": "candidate_not_promoted_user_approval_required",
                        "search_profile_id": "qdrift_samples_rng_v1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(
        _native_comparator_case(tmp_path),
        manifest_path=settings_manifest,
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_qdrift",
        output_dir=tmp_path / "qdrift_class_settings",
    ).to_dict()

    assert row["status"] == "completed"
    assert row["metrics"]["samples_per_interval"] == 3
    assert row["metrics"]["rng_seed"] == 123
    assert row["provenance"]["settings_source"] == "unit_all_algorithm_candidate"
    assert row["provenance"]["class_settings_candidate_only_not_promoted"] is True
    assert row["provenance"]["class_settings_search_profile_id"] == "qdrift_samples_rng_v1"
    assert row["provenance"]["class_tuned_result_locked"] is False
    raw = json.loads((tmp_path / "qdrift_class_settings" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["parameter_manifest"]["tuning_provenance"]["settings_source"] == "unit_all_algorithm_candidate"


def test_required_comparator_class_settings_missing_fails_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def fake_load(*args, **kwargs):
        nonlocal called
        called = True
        return _single_qubit_runtime_input()

    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", fake_load)
    settings_manifest = tmp_path / "class_settings.json"
    settings_manifest.write_text(
        json.dumps(
            {
                "schema": "dynamics_class_settings_lock_manifest_v1",
                "lock_status": "candidate_not_promoted",
                "required_algorithm_settings": [
                    {"algorithm_id": "dyn_qdrift", "settings_kind": "comparator"}
                ],
                "settings": [],
            }
        ),
        encoding="utf-8",
    )
    case = with_class_settings_lock_manifest(
        _native_comparator_case(tmp_path),
        manifest_path=settings_manifest,
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_qdrift",
        output_dir=tmp_path / "qdrift_missing_required_settings",
    ).to_dict()

    assert called is False
    assert row["status"] == "failed"
    assert "missing required all-algorithm class entries" in row["reason"]


def test_native_comparator_honors_declared_driven_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _single_qubit_runtime_input()
    h_poly = PauliPolynomial("JW")
    h_poly.add_term(PauliTerm(2, ps="ze", pc=0.7))
    h_poly._reduce()
    runtime.h_poly = h_poly
    runtime.psi_ref = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex)
    runtime.psi_initial = runtime.psi_ref.copy()
    runtime.resolved_problem = SimpleNamespace(
        family_key="hubbard",
        hamiltonian=h_poly,
        request=SimpleNamespace(num_sites=1, ordering="blocked"),
    )
    monkeypatch.setattr(rows_mod, "load_scaffold_runtime_input", lambda *args, **kwargs: runtime)
    case = _native_comparator_case(
        tmp_path,
        metadata={
            "enable_drive": True,
            "drive": {
                "enable_drive": True,
                "A": 0.1,
                "omega": 1.0,
                "tbar": 0.5,
                "phi": 0.0,
                "pattern": "staggered",
                "time_sampling": "midpoint",
            },
        },
    )

    row = rows_mod.run_generic_dynamics_row(
        case=case,
        algorithm_id="dyn_qdrift",
        output_dir=tmp_path / "driven",
    ).to_dict()

    assert row["status"] == "completed"
    assert row["qpu_faithful"] is True
    assert (tmp_path / "driven" / "rows.json").exists()
    raw = json.loads((tmp_path / "driven" / "raw_payload.json").read_text(encoding="utf-8"))
    assert raw["parameter_manifest"]["time_dependence"] == "driven_hamiltonian"
    assert raw["parameter_manifest"]["drive_included"] is True
    assert raw["parameter_manifest"]["drive_profile"]["A"] == pytest.approx(0.1)


def test_run_single_hh_legacy_wrapper_command_still_dispatches_hh_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(legacy_native.subprocess, "run", fake_subprocess_run)

    payload = bench.run_single(
        family="hh",
        case_id="hh_l2_t8_anchor_v1",
        algorithm_id="dyn_exact_reference",
        output_dir=tmp_path,
    )

    assert payload["status"] == "ok"
    assert captured["cmd"][:3] == [sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_exact_reference_benchmark"]
    assert captured["cmd"][captured["cmd"].index("--case-id") + 1] == "hh_l2_t8_anchor_v1"
    assert (tmp_path / "command.json").exists()


def test_manifest_build_forwards_class_settings_and_seed_lock_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "seed.json"
    artifact.write_text('{"settings": {"problem": "hubbard"}}\n', encoding="utf-8")
    case_manifest = tmp_path / "cases.json"
    case_manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "hubbard_locked_seed",
                        "family": "hubbard",
                        "table_class": "fermionic_lattice",
                        "artifact_json": str(artifact),
                        "metadata": {
                            "seed_lock": {
                                "same_seed_comparator_group_id": "hubbard_A0p2_same_seed"
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    settings_manifest = tmp_path / "class_settings.json"
    settings_manifest.write_text(
        json.dumps(
            {
                "schema": "dynamics_class_settings_lock_manifest_v1",
                "lock_status": "locked",
                "settings": [
                    {
                        "tuning_class": "fermionic",
                        "algorithm_id": "dyn_controller_full",
                        "settings_kind": "controller",
                        "class_tuned_result_locked": True,
                        "settings_payload": {"miss_threshold": 0.3},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    jobs = bench.build_dynamics_jobs(
        output_root=tmp_path / "out",
        families=("hubbard",),
        algorithm_ids=("dyn_fixed_mclachlan",),
        include_skipped=True,
        case_manifest=case_manifest,
        class_settings_manifest=settings_manifest,
        require_locked_class_settings=True,
    )

    assert len(jobs) == 1
    command = list(jobs[0].command)
    assert "--class-settings-manifest" in command
    assert str(settings_manifest) in command
    assert "--require-locked-class-settings" in command
    assert jobs[0].metadata["same_seed_comparator_group_id"] == "hubbard_A0p2_same_seed"
    assert jobs[0].metadata["static_seed_artifact_sha256"]
