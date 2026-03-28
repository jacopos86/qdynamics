from __future__ import annotations

from pathlib import Path
import queue as pyqueue
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
                            "backend_name": "FakeMarrakesh",
                            "optimization_level": 1,
                            "seed_transpiler": 0,
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
            subject_kind="hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
            structure_locked=True,
            term_order_id="source_order",
            operator_count=6,
            runtime_term_count=6,
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
    call_counts = {"ideal": 0}

    def _fake_ideal_eval_locked(**kwargs):
        call_counts["ideal"] += 1
        return {
            "ideal_mean": 0.25,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        }

    def _fake_noisy_eval_locked(**kwargs):
        opt = int(kwargs["transpile_optimization_level"])
        seed_trans = int(kwargs["seed_transpiler"])
        delta, two_qubit, depth, size = score_map[(opt, seed_trans)]
        noisy_mean = 0.25 + float(delta)
        return {
            "noisy_mean": noisy_mean,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
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

    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        _fake_ideal_eval_locked,
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_noisy_energy",
        _fake_noisy_eval_locked,
    )

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
    assert payload["baseline_candidate"]["requested_seed_transpiler"] == 7
    assert payload["baseline_candidate"]["requested_transpile_optimization_level"] == 1
    assert payload["baseline_candidate"]["compile_observation"]["available"] is True
    assert payload["baseline_candidate"]["compile_observation"]["matches_requested"] is True
    assert payload["best_candidate"]["compile_observation"]["matches_requested"] is True
    assert payload["ranking"]["candidate_labels_ranked"][0] == "opt2_seed1"
    assert payload["ranking"]["best_vs_baseline_delta_abs_improvement"] == pytest.approx(0.4)
    assert payload["ranking"]["best_vs_baseline_two_qubit_delta"] == -2
    assert payload["ranking"]["best_vs_baseline_depth_delta"] == -9
    assert payload["noise_config"]["mitigation"]["local_readout_strategy"] == "mthree"
    assert call_counts["ideal"] == 1


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
        backend_name="FakeMarrakesh",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        baseline_transpile_optimization_level=1,
        baseline_seed_transpiler=0,
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
        (1, 0): (0.9, 25, 63, 108),
        (1, 1): (0.4, 22, 58, 101),
        (2, 0): (0.4, 21, 57, 99),
        (2, 1): (0.5, 20, 56, 98),
    }
    call_counts = {"ideal": 0}

    def _fake_ideal_eval_locked(**kwargs):
        call_counts["ideal"] += 1
        return {
            "ideal_mean": 0.25,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        }

    def _fake_noisy_eval_locked(**kwargs):
        opt = int(kwargs["transpile_optimization_level"])
        seed_trans = int(kwargs["seed_transpiler"])
        delta, two_qubit, depth, size = score_map[(opt, seed_trans)]
        noisy_mean = 0.25 + float(delta)
        return {
            "noisy_mean": noisy_mean,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "fake_backend.run(counts)",
                "backend_name": "FakeMarrakesh",
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

    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        _fake_ideal_eval_locked,
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_noisy_energy",
        _fake_noisy_eval_locked,
    )

    payload = report._run_imported_fixed_scaffold_compile_control_scout(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        baseline_transpile_optimization_level=1,
        baseline_seed_transpiler=0,
        scout_transpile_optimization_levels=[1, 2],
        scout_seed_transpilers=[0, 1],
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["route"] == "fixed_scaffold_compile_control_scout"
    assert payload["subject_kind"] == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert payload["term_order_id"] == "source_order"
    assert payload["candidate_counts"] == {"total": 4, "successful": 4, "failed": 0}
    assert payload["baseline_candidate"]["label"] == "opt1_seed0"
    assert payload["best_candidate"]["label"] == "opt2_seed0"
    assert payload["best_candidate"]["compiled_two_qubit_count"] == 21
    assert payload["artifact_compile_recommendation"] == {
        "backend_name": "FakeMarrakesh",
        "optimization_level": 1,
        "seed_transpiler": 0,
    }
    assert payload["baseline_candidate"]["requested_backend_name"] == "FakeMarrakesh"
    assert payload["baseline_candidate"]["requested_seed_transpiler"] == 0
    assert payload["baseline_candidate"]["requested_transpile_optimization_level"] == 1
    assert payload["baseline_candidate"]["compile_observation"]["available"] is True
    assert payload["baseline_candidate"]["compile_observation"]["matches_requested"] is True
    assert payload["best_candidate"]["compile_observation"]["matches_requested"] is True
    assert payload["baseline_compile_observation"]["matches_requested"] is True
    assert payload["best_candidate_compile_observation"]["matches_requested"] is True
    assert payload["ranking"]["candidate_labels_ranked"][0] == "opt2_seed0"
    assert payload["ranking"]["best_vs_baseline_delta_abs_improvement"] == pytest.approx(0.5)
    assert payload["ranking"]["best_vs_baseline_two_qubit_delta"] == -4
    assert call_counts["ideal"] == 1


def test_fixed_scaffold_compile_control_scout_isolated_timeout_reports_partial_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_payload = {
        "success": False,
        "available": True,
        "route": "fixed_scaffold_compile_control_scout",
        "reason": "in_progress",
        "artifact_json": "artifacts/json/example.json",
        "candidate_counts": {"total": 10, "completed": 1, "successful": 1, "failed": 0},
        "best_candidate": {
            "label": "opt1_seed0",
            "delta_mean": 0.9,
            "compiled_two_qubit_count": 14,
            "compiled_depth": 48,
            "compile_observation": {"matches_requested": True},
        },
        "best_candidate_compile_observation": {"matches_requested": True},
        "baseline_candidate": {
            "label": "opt1_seed0",
            "compile_observation": {"matches_requested": True},
        },
        "baseline_compile_observation": {"matches_requested": True},
        "artifact_compile_recommendation": {
            "backend_name": "FakeMarrakesh",
            "optimization_level": 1,
            "seed_transpiler": 0,
        },
        "last_candidate_label": "opt1_seed0",
        "last_candidate_index": 0,
        "elapsed_s": 12.0,
    }

    class _FakeQueue:
        def __init__(self, messages):
            self._messages = list(messages)

        def get_nowait(self):
            if not self._messages:
                raise pyqueue.Empty
            return self._messages.pop(0)

        def empty(self):
            return not self._messages

        def get(self):
            return self.get_nowait()

    class _FakeProcess:
        def __init__(self):
            self.exitcode = None
            self._alive = True

        def start(self) -> None:
            return None

        def join(self, timeout=None) -> None:
            return None

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False
            self.exitcode = -15

    fake_process = _FakeProcess()
    fake_queue = _FakeQueue(
        [
            {
                "kind": "progress",
                "payload": {
                    "event": "compile_control_scout_candidate_completed",
                    "partial_payload": partial_payload,
                },
            }
        ]
    )

    class _FakeContext:
        def Queue(self):
            return fake_queue

        def Process(self, *args, **kwargs):
            return fake_process

    class _Clock:
        def __init__(self):
            self._values = [0.0, 0.1, 1.1]

        def __call__(self):
            if len(self._values) > 1:
                return self._values.pop(0)
            return self._values[0]

    monkeypatch.setattr(report.mp, "get_context", lambda mode: _FakeContext())
    monkeypatch.setattr(report.time, "monotonic", _Clock())

    payload = report._run_imported_fixed_scaffold_compile_control_scout_mode_isolated(
        kwargs={"artifact_json": "ignored"},
        timeout_s=1,
    )

    assert payload["success"] is False
    assert payload["reason"] == "timeout_after_1s"
    assert payload["candidate_counts"] == {"total": 10, "completed": 1, "successful": 1, "failed": 0}
    assert payload["best_candidate"]["label"] == "opt1_seed0"
    assert payload["baseline_compile_observation"]["matches_requested"] is True
    assert payload["artifact_compile_recommendation"]["backend_name"] == "FakeMarrakesh"
    assert payload["last_progress_event"] == "compile_control_scout_candidate_completed"
    assert payload["elapsed_s"] == pytest.approx(1.1)


def test_fixed_scaffold_compile_control_scout_isolated_nonzero_exit_reports_partial_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial_payload = {
        "success": False,
        "available": True,
        "route": "fixed_scaffold_compile_control_scout",
        "reason": "in_progress",
        "artifact_json": "artifacts/json/example.json",
        "candidate_counts": {"total": 10, "completed": 0, "successful": 0, "failed": 0},
        "last_candidate_label": "opt1_seed0",
        "last_candidate_index": 0,
        "elapsed_s": 12.0,
    }

    class _FakeQueue:
        def __init__(self, messages):
            self._messages = list(messages)

        def get_nowait(self):
            if not self._messages:
                raise pyqueue.Empty
            return self._messages.pop(0)

        def empty(self):
            return not self._messages

        def get(self):
            return self.get_nowait()

    class _FakeProcess:
        def __init__(self):
            self.exitcode = -6
            self._alive = True

        def start(self) -> None:
            return None

        def join(self, timeout=None) -> None:
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

    fake_queue = _FakeQueue(
        [
            {
                "kind": "progress",
                "payload": {
                    "event": "compile_control_scout_candidate_started",
                    "partial_payload": partial_payload,
                },
            }
        ]
    )

    class _FakeContext:
        def Queue(self):
            return fake_queue

        def Process(self, *args, **kwargs):
            return _FakeProcess()

    class _Clock:
        def __init__(self):
            self._values = [0.0, 0.1, 0.2]

        def __call__(self):
            if len(self._values) > 1:
                return self._values.pop(0)
            return self._values[0]

    monkeypatch.setattr(report.mp, "get_context", lambda mode: _FakeContext())
    monkeypatch.setattr(report.time, "monotonic", _Clock())

    payload = report._run_imported_fixed_scaffold_compile_control_scout_mode_isolated(
        kwargs={"artifact_json": "ignored"},
        timeout_s=30,
    )

    assert payload["success"] is False
    assert payload["reason"] == "subprocess_nonzero_exit"
    assert payload["exitcode"] == -6
    assert payload["candidate_counts"] == {"total": 10, "completed": 0, "successful": 0, "failed": 0}
    assert payload["last_candidate_label"] == "opt1_seed0"
    assert payload["last_progress_event"] == "compile_control_scout_candidate_started"
    assert payload["elapsed_s"] == pytest.approx(0.2)
