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
    logical_parameter_count = 7
    runtime_parameter_count = 7


def _locked_lean_ctx(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], str, None]:
    return (
        {
            "path": tmp_path / "lean.json",
            "source_kind": "direct_adapt_artifact",
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "circuit": object(),
        },
        {
            "source": "hf",
            "handoff_state_kind": "reference_state",
        },
        "pareto_lean_l2",
        None,
    )


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
            "circuit": object(),
            "payload": {
                "adapt_vqe": {
                    "fixed_scaffold_metadata": {
                        "compile_recommendation": {
                            "backend_name": "FakeNighthawk",
                            "optimization_level": 2,
                            "seed_transpiler": 7,
                        }
                    }
                }
            },
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


def test_fixed_lean_compile_control_scout_requires_local_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_lean_context",
        lambda artifact_json, nonlean_reason: _locked_lean_ctx(tmp_path),
    )

    payload = report._run_imported_fixed_lean_compile_control_scout(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeHeron",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        baseline_transpile_optimization_level=1,
        baseline_seed_transpiler=7,
        scout_transpile_optimization_levels=[1, 2],
        scout_seed_transpilers=[0, 1],
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_lean_compile_control_scout_requires_local_fake_backend"


def test_fixed_lean_compile_control_scout_ranks_successful_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_lean_context",
        lambda artifact_json, nonlean_reason: _locked_lean_ctx(tmp_path),
    )

    score_map = {
        (1, 7): (0.9, 20, 70, 100),
        (1, 0): (0.5, 22, 71, 102),
        (1, 1): (0.7, 21, 72, 101),
        (2, 0): (0.5, 19, 62, 92),
        (2, 1): (0.5, 18, 61, 91),
    }

    def _fake_eval_locked(**kwargs):
        opt = int(kwargs["transpile_optimization_level"])
        seed_trans = int(kwargs["seed_transpiler"])
        delta, two_qubit, depth, size = score_map[(opt, seed_trans)]
        ideal_mean = 0.25
        noisy_mean = ideal_mean + float(delta)
        return {
            "noisy_mean": noisy_mean,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
            "ideal_mean": ideal_mean,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": float(delta),
            "delta_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "fake_backend.run(counts)",
                "backend_name": "FakeHeron",
                "using_fake_backend": True,
                "details": {
                    "transpile_seed": int(seed_trans),
                    "transpile_optimization_level": int(opt),
                    "compiled_two_qubit_count": int(two_qubit),
                    "compiled_cx_count": int(two_qubit),
                    "compiled_ecr_count": 0,
                    "compiled_depth": int(depth),
                    "compiled_size": int(size),
                    "compiled_num_qubits": 6,
                    "layout_physical_qubits": [0, 1, 2, 3, 4, 5],
                    "compiled_op_counts": {},
                },
            },
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy", _fake_eval_locked)

    payload = report._run_imported_fixed_lean_compile_control_scout(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeHeron",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        baseline_transpile_optimization_level=1,
        baseline_seed_transpiler=7,
        scout_transpile_optimization_levels=[1, 2],
        scout_seed_transpilers=[0, 1],
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["route"] == "fixed_lean_compile_control_scout"
    assert payload["candidate_counts"] == {"total": 5, "successful": 5, "failed": 0}
    assert payload["baseline_candidate"]["label"] == "opt1_seed7"
    assert payload["best_candidate"]["label"] == "opt2_seed1"
    assert payload["best_candidate"]["compiled_two_qubit_count"] == 18
    assert payload["ranking"]["candidate_labels_ranked"][0] == "opt2_seed1"
    assert payload["ranking"]["best_vs_baseline_delta_abs_improvement"] == pytest.approx(0.4)
    assert payload["ranking"]["best_vs_baseline_two_qubit_delta"] == -2
    assert payload["ranking"]["best_vs_baseline_depth_delta"] == -9
    assert payload["noise_config"]["mitigation"]["local_readout_strategy"] == "mthree"


def test_fixed_scaffold_compile_control_scout_requires_local_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )

    payload = report._run_imported_fixed_scaffold_compile_control_scout(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeNighthawk",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        baseline_transpile_optimization_level=2,
        baseline_seed_transpiler=7,
        scout_transpile_optimization_levels=[1, 2],
        scout_seed_transpilers=[0, 1],
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_scaffold_compile_control_scout_requires_local_fake_backend"


def test_fixed_scaffold_compile_control_scout_reports_metadata_and_ranking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )

    score_map = {
        (2, 7): (0.9, 25, 63, 108),
        (1, 0): (0.6, 23, 60, 103),
        (1, 1): (0.4, 22, 58, 101),
        (2, 0): (0.4, 21, 57, 99),
        (2, 1): (0.5, 20, 56, 98),
    }

    def _fake_eval_locked(**kwargs):
        opt = int(kwargs["transpile_optimization_level"])
        seed_trans = int(kwargs["seed_transpiler"])
        delta, two_qubit, depth, size = score_map[(opt, seed_trans)]
        ideal_mean = 0.25
        noisy_mean = ideal_mean + float(delta)
        return {
            "noisy_mean": noisy_mean,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
            "ideal_mean": ideal_mean,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": float(delta),
            "delta_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "fake_backend.run(counts)",
                "backend_name": "FakeNighthawk",
                "using_fake_backend": True,
                "details": {
                    "transpile_seed": int(seed_trans),
                    "transpile_optimization_level": int(opt),
                    "compiled_two_qubit_count": int(two_qubit),
                    "compiled_cx_count": 0,
                    "compiled_ecr_count": int(two_qubit),
                    "compiled_depth": int(depth),
                    "compiled_size": int(size),
                    "compiled_num_qubits": 6,
                    "layout_physical_qubits": [0, 1, 2, 3, 4, 5],
                    "compiled_op_counts": {},
                },
            },
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy", _fake_eval_locked)

    payload = report._run_imported_fixed_scaffold_compile_control_scout(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeNighthawk",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        baseline_transpile_optimization_level=2,
        baseline_seed_transpiler=7,
        scout_transpile_optimization_levels=[1, 2],
        scout_seed_transpilers=[0, 1],
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["route"] == "fixed_scaffold_compile_control_scout"
    assert payload["subject_kind"] == "hh_nighthawk_gate_pruned_7term_v1"
    assert payload["term_order_id"] == "source_order"
    assert payload["candidate_counts"] == {"total": 5, "successful": 5, "failed": 0}
    assert payload["baseline_candidate"]["label"] == "opt2_seed7"
    assert payload["best_candidate"]["label"] == "opt2_seed0"
    assert payload["best_candidate"]["compiled_two_qubit_count"] == 21
    assert payload["artifact_compile_recommendation"] == {
        "backend_name": "FakeNighthawk",
        "optimization_level": 2,
        "seed_transpiler": 7,
    }
    assert payload["ranking"]["candidate_labels_ranked"][0] == "opt2_seed0"
    assert payload["ranking"]["best_vs_baseline_delta_abs_improvement"] == pytest.approx(0.5)
    assert payload["ranking"]["best_vs_baseline_two_qubit_delta"] == -4
