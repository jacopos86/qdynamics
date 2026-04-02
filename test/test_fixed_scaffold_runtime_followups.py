from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.fixed_scaffold_runtime_followups import (
    build_fixed_scaffold_rerun_plan,
    reconstruct_fixed_scaffold_runtime_recovery,
)
from pipelines.exact_bench.noise_oracle_runtime import (
    RuntimeJobRecord,
    SubmittedRuntimeJobError,
    _fetch_runtime_job_record,
    _run_estimator_job,
)


def test_run_estimator_job_emits_runtime_events() -> None:
    class _Backend:
        name = "ibm_test"

    class _Job:
        def job_id(self):
            return "job-123"

        def status(self):
            return "DONE"

        @property
        def creation_date(self):
            return datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)

        def backend(self):
            return _Backend()

        @property
        def session_id(self):
            return None

        def usage(self):
            return 12

        def metrics(self):
            return {
                "timestamps": {
                    "created": "2026-03-23T12:00:00Z",
                    "running": "2026-03-23T12:00:01Z",
                    "finished": "2026-03-23T12:00:02Z",
                },
                "usage": {"quantum_seconds": 12},
            }

        def result(self):
            class _Data:
                evs = [0.125]

            class _Pub:
                data = _Data()

            return [_Pub()]

    class _Estimator:
        def run(self, pub):
            return _Job()

    events: list[dict] = []
    qc = QuantumCircuit(1)
    op = SparsePauliOp.from_list([("Z", 1.0)])
    result = _run_estimator_job(
        _Estimator(),
        qc,
        op,
        repeat_index=2,
        job_observer=events.append,
        job_context={"call_index": 7},
    )

    assert result.expectation_value == pytest.approx(0.125)
    assert len(result.job_records) == 1
    assert [evt["event"] for evt in events] == ["submitted", "completed"]
    assert events[0]["call_index"] == 7
    assert events[1]["job"]["job_id"] == "job-123"


def test_reconstruct_fixed_scaffold_runtime_recovery_orders_jobs_and_picks_best(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )

    table = {
        "job-a": RuntimeJobRecord(
            job_id="job-a",
            repeat_index=0,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:03Z",
            completed_utc="2026-03-23T12:00:04Z",
            expectation_value=0.5,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
        "job-b": RuntimeJobRecord(
            job_id="job-b",
            repeat_index=0,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:01Z",
            completed_utc="2026-03-23T12:00:02Z",
            expectation_value=0.3,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
        "job-c": RuntimeJobRecord(
            job_id="job-c",
            repeat_index=0,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:05Z",
            completed_utc="2026-03-23T12:00:06Z",
            expectation_value=0.4,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
    }
    monkeypatch.setattr(
        "pipelines.exact_bench.fixed_scaffold_runtime_followups._fetch_runtime_job_record",
        lambda job_id, require_result: table[str(job_id)],
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        runtime_job_ids=["job-a", "job-b", "job-c"],
        output_json=tmp_path / "recovery.json",
    )

    assert payload["recovery_granularity"] == "runtime_job"
    assert payload["summary"]["trace_rows_total"] == 3
    assert payload["summary"]["objective_calls_total"] is None
    assert payload["best_so_far"]["trace_index"] == 1
    assert payload["best_so_far"]["call_index"] is None
    assert payload["best_so_far"]["energy_noisy_mean"] == pytest.approx(0.3)


def test_reconstruct_fixed_scaffold_runtime_recovery_reads_nested_staged_noisy_replay_payload(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )
    recovery_source = tmp_path / "staged.json"
    recovery_source.write_text(
        """
        {
          "fixed_scaffold_noisy_replay": {
            "objective_trace": [
              {
                "call_index": 1,
                "status": "completed",
                "elapsed_s": 0.25,
                "theta_runtime": [0.1, -0.2],
                "theta_logical": [0.1, -0.2],
                "energy_noisy_mean": 0.5,
                "energy_noisy_stderr": 0.02,
                "n_samples": 1
              },
              {
                "call_index": 2,
                "status": "completed",
                "elapsed_s": 0.75,
                "theta_runtime": [0.05, -0.15],
                "theta_logical": [0.05, -0.15],
                "energy_noisy_mean": 0.3,
                "energy_noisy_stderr": 0.01,
                "n_samples": 1
              }
            ],
            "runtime_job_ids": []
          }
        }
        """,
        encoding="utf-8",
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        recovery_source_json=recovery_source,
        output_json=tmp_path / "recovery.json",
    )

    assert payload["recovery_granularity"] == "objective_call"
    assert payload["reconstructed_from_runtime_jobs"] is False
    assert payload["summary"]["trace_rows_total"] == 2
    assert payload["summary"]["trace_rows_completed"] == 2
    assert payload["best_so_far"]["call_index"] == 2
    assert payload["best_so_far"]["energy_noisy_mean"] == pytest.approx(0.3)
    assert payload["objective_trace"][1]["raw_values"] == pytest.approx([0.3])
    assert (
        "Recovered from fixed_scaffold_noisy_replay embedded in staged workflow output."
        in payload["notes"]
    )


def test_reconstruct_fixed_scaffold_runtime_recovery_reads_direct_noisy_replay_payload(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )
    recovery_source = tmp_path / "direct_replay.json"
    recovery_source.write_text(
        """
        {
          "objective_trace": [
            {
              "call_index": 1,
              "status": "completed",
              "theta_runtime": [0.1, -0.2],
              "theta_logical": [0.1, -0.2],
              "energy_noisy_mean": 0.45,
              "energy_noisy_stderr": 0.02
            },
            {
              "call_index": 2,
              "status": "completed",
              "theta_runtime": [0.05, -0.15],
              "theta_logical": [0.05, -0.15],
              "energy_noisy_mean": 0.35,
              "energy_noisy_stderr": 0.01
            }
          ],
          "runtime_job_ids": []
        }
        """,
        encoding="utf-8",
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        recovery_source_json=recovery_source,
    )

    assert payload["recovery_granularity"] == "objective_call"
    assert payload["summary"]["trace_rows_total"] == 2
    assert payload["best_so_far"]["call_index"] == 2
    assert payload["best_so_far"]["energy_noisy_mean"] == pytest.approx(0.35)


def test_reconstruct_fixed_scaffold_runtime_recovery_explicit_runtime_job_ids_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )
    recovery_source = tmp_path / "direct_replay.json"
    recovery_source.write_text(
        """
        {
          "objective_trace": [
            {
              "call_index": 1,
              "status": "completed",
              "theta_runtime": [0.1, -0.2],
              "theta_logical": [0.1, -0.2],
              "energy_noisy_mean": 0.05,
              "energy_noisy_stderr": 0.01
            }
          ],
          "runtime_job_ids": []
        }
        """,
        encoding="utf-8",
    )

    table = {
        "job-a": RuntimeJobRecord(
            job_id="job-a",
            repeat_index=0,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:01Z",
            completed_utc="2026-03-23T12:00:02Z",
            expectation_value=0.4,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
        "job-b": RuntimeJobRecord(
            job_id="job-b",
            repeat_index=0,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:03Z",
            completed_utc="2026-03-23T12:00:04Z",
            expectation_value=0.3,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
    }
    monkeypatch.setattr(
        "pipelines.exact_bench.fixed_scaffold_runtime_followups._fetch_runtime_job_record",
        lambda job_id, require_result: table[str(job_id)],
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        recovery_source_json=recovery_source,
        runtime_job_ids=["job-a", "job-b"],
    )

    assert payload["recovery_granularity"] == "objective_call"
    assert payload["objective_trace"][0]["call_index"] == 1
    assert payload["summary"]["objective_calls_total"] == 1
    assert payload["best_so_far"]["call_index"] is None
    assert payload["best_so_far"]["energy_noisy_mean"] == pytest.approx(0.3)
    assert (
        "Explicit runtime_job_ids took precedence over any embedded recovery-source job ids or objective trace."
        in payload["notes"]
    )


def test_reconstruct_fixed_scaffold_runtime_recovery_explicit_pending_job_blocks_source_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )
    recovery_source = tmp_path / "direct_replay.json"
    recovery_source.write_text(
        """
        {
          "objective_trace": [
            {
              "call_index": 1,
              "status": "completed",
              "theta_runtime": [0.1, -0.2],
              "theta_logical": [0.1, -0.2],
              "energy_noisy_mean": 0.05,
              "energy_noisy_stderr": 0.01
            }
          ],
          "runtime_job_ids": []
        }
        """,
        encoding="utf-8",
    )

    pending = RuntimeJobRecord(
        job_id="job-pending",
        repeat_index=0,
        call_path="lookup",
        status="QUEUED",
        created_utc="2026-03-23T12:00:01Z",
        completed_utc=None,
        expectation_value=None,
        backend_name="ibm_kingston",
        usage_quantum_seconds=0.0,
    )
    monkeypatch.setattr(
        "pipelines.exact_bench.fixed_scaffold_runtime_followups._fetch_runtime_job_record",
        lambda job_id, require_result: pending,
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        recovery_source_json=recovery_source,
        runtime_job_ids=["job-pending"],
    )

    assert payload["objective_trace"][0]["call_index"] == 1
    assert payload["objective_trace"][0]["status"] == "partial_pending"
    assert payload["objective_trace"][0]["energy_noisy_mean"] is None
    assert payload["best_so_far"] is None


def test_reconstruct_fixed_scaffold_runtime_recovery_explicit_job_override_preserves_row_grouping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )
    recovery_source = tmp_path / "direct_replay.json"
    recovery_source.write_text(
        """
        {
          "objective_trace": [
            {
              "call_index": 1,
              "status": "completed",
              "theta_runtime": [0.1, -0.2],
              "theta_logical": [0.1, -0.2],
              "energy_noisy_mean": 0.9,
              "energy_noisy_stderr": 0.2,
              "runtime_jobs": [{"job_id": "source-a"}, {"job_id": "source-b"}]
            }
          ],
          "runtime_job_ids": []
        }
        """,
        encoding="utf-8",
    )

    table = {
        "job-a": RuntimeJobRecord(
            job_id="job-a",
            repeat_index=0,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:01Z",
            completed_utc="2026-03-23T12:00:02Z",
            expectation_value=0.4,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
        "job-b": RuntimeJobRecord(
            job_id="job-b",
            repeat_index=1,
            call_path="lookup",
            status="DONE",
            created_utc="2026-03-23T12:00:03Z",
            completed_utc="2026-03-23T12:00:04Z",
            expectation_value=0.6,
            backend_name="ibm_kingston",
            usage_quantum_seconds=12.0,
        ),
    }
    monkeypatch.setattr(
        "pipelines.exact_bench.fixed_scaffold_runtime_followups._fetch_runtime_job_record",
        lambda job_id, require_result: table[str(job_id)],
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        recovery_source_json=recovery_source,
        runtime_job_ids=["job-a", "job-b"],
    )

    assert payload["summary"]["trace_rows_total"] == 1
    assert payload["objective_trace"][0]["call_index"] == 1
    assert payload["objective_trace"][0]["n_samples"] == 2
    assert payload["objective_trace"][0]["energy_noisy_mean"] == pytest.approx(0.5)
    assert len(payload["objective_trace"][0]["runtime_jobs"]) == 2


def test_reconstruct_fixed_scaffold_runtime_recovery_explicit_failed_job_marks_row_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        """
        {
          "ground_state": {"exact_energy": 0.1},
          "adapt_vqe": {
            "energy": 0.2,
            "parameterization": {"runtime_parameter_count": 6}
          }
        }
        """,
        encoding="utf-8",
    )
    recovery_source = tmp_path / "direct_replay.json"
    recovery_source.write_text(
        """
        {
          "objective_trace": [
            {
              "call_index": 1,
              "status": "completed",
              "theta_runtime": [0.1, -0.2],
              "theta_logical": [0.1, -0.2],
              "energy_noisy_mean": 0.05,
              "energy_noisy_stderr": 0.01
            }
          ],
          "runtime_job_ids": []
        }
        """,
        encoding="utf-8",
    )

    failed = RuntimeJobRecord(
        job_id="job-failed",
        repeat_index=0,
        call_path="lookup",
        status="FAILED",
        created_utc="2026-03-23T12:00:01Z",
        completed_utc="2026-03-23T12:00:02Z",
        expectation_value=None,
        backend_name="ibm_kingston",
        usage_quantum_seconds=0.0,
    )
    monkeypatch.setattr(
        "pipelines.exact_bench.fixed_scaffold_runtime_followups._fetch_runtime_job_record",
        lambda job_id, require_result: failed,
    )

    payload = reconstruct_fixed_scaffold_runtime_recovery(
        artifact_json=candidate,
        recovery_source_json=recovery_source,
        runtime_job_ids=["job-failed"],
    )

    assert payload["objective_trace"][0]["status"] == "failed"
    assert payload["objective_trace"][0]["energy_noisy_mean"] is None
    assert payload["best_so_far"] is None


def test_run_estimator_job_preserves_submitted_job_record_on_failure() -> None:
    class _Backend:
        name = "ibm_test"

    class _Job:
        def job_id(self):
            return "job-fail"

        def status(self):
            return "RUNNING"

        @property
        def creation_date(self):
            return datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)

        def backend(self):
            return _Backend()

        @property
        def session_id(self):
            return None

        def usage(self):
            return 0

        def metrics(self):
            return {}

        def result(self):
            raise RuntimeError("boom")

    class _Estimator:
        def run(self, pub):
            return _Job()

    events: list[dict] = []
    qc = QuantumCircuit(1)
    op = SparsePauliOp.from_list([("Z", 1.0)])
    with pytest.raises(SubmittedRuntimeJobError) as excinfo:
        _run_estimator_job(
            _Estimator(),
            qc,
            op,
            repeat_index=0,
            job_observer=events.append,
            job_context={"call_index": 2},
        )
    assert excinfo.value.record.job_id == "job-fail"
    assert [evt["event"] for evt in events] == ["submitted", "failed"]


def test_fetch_runtime_job_record_skips_result_for_pending_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Job:
        def __init__(self) -> None:
            self.result_called = False

        def status(self):
            return "QUEUED"

        @property
        def creation_date(self):
            return datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)

        def result(self):
            self.result_called = True
            raise AssertionError("result() should not be called for queued jobs")

        def metrics(self):
            return {}

        def usage(self):
            return None

        def backend(self):
            return None

        @property
        def session_id(self):
            return None

    job = _Job()

    class _Svc:
        def job(self, job_id: str):
            return job

    import qiskit_ibm_runtime

    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", lambda: _Svc())
    rec = _fetch_runtime_job_record("job-queued", require_result=True)
    assert rec.status == "QUEUED"
    assert rec.expectation_value is None
    assert "Skipped result lookup" in str(rec.error)
    assert job.result_called is False


def test_build_fixed_scaffold_rerun_plan_current_artifacts() -> None:
    payload = build_fixed_scaffold_rerun_plan(
        candidate_artifact_json=REPO_ROOT / "artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json",
        recovery_json=REPO_ROOT / "artifacts/json/hh_kingston_6term_spsa48_runtime_partial_recovery_20260323.json",
    )

    assert payload["structural_diff"]["omitted_runtime_terms_exyz"] == ["eyezee"]
    assert payload["noise_evidence"]["dominant_residual_noise_source"] == "gate_stateprep"
    assert payload["noise_evidence"]["readout_not_primary_limit"] is True
    assert payload["budget_plan"]["candidate_screen"]["screen_total_runtime_jobs"] == 15
    assert payload["budget_plan"]["anchor_screen"]["screen_total_runtime_jobs"] == 17
    assert payload["budget_plan"]["backend_selection_required"] is True
    assert str(payload["budget_plan"]["recommended_next_submission"]).startswith(
        "select_backend_then_anchor_7term_energy_only_fixedtheta_baseline"
    )
