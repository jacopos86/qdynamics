from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pipelines.reporting.paper_i_run_summary import (
    PaperIAcceptedError,
    PaperIAlgorithmicWork,
    PaperIAppendMatchedObservation,
    PaperIEffectivePlateauObservation,
    PaperIRunProvenance,
    PaperIRunSummary,
    PaperIWorkComponents,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page16_insertion_comparators_20260812.py"
)
SUPERVISOR_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "supervise_local_page16_insertion_comparator_waves_20260812.py"
)


def _load_runner():
    name = "paper_i_page16_local_insertion_comparator_runner_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_supervisor():
    name = "paper_i_page16_local_insertion_comparator_supervisor_test"
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _typed_round30_summary() -> PaperIRunSummary:
    exact_energy = -1.0
    errors = tuple(0.5 / round_index for round_index in range(1, 31))
    trace = tuple(
        PaperIAcceptedError(
            controller_round=round_index,
            active_ansatz_depth=round_index,
            accepted_energy=exact_energy + error,
            exact_same_cutoff_energy=exact_energy,
            absolute_energy_error=error,
            projective_state_fingerprint=f"state-{round_index}",
            checkpoint_sha256=f"{round_index:064x}",
        )
        for round_index, error in enumerate(errors, start=1)
    )
    work = PaperIAlgorithmicWork(
        components=PaperIWorkComponents(
            n_h_outer=1,
            n_h_refit=2,
            n_grad=3,
            n_metric=4,
        ),
        s_alg=10,
    )
    return PaperIRunSummary(
        accepted_error_trace=trace,
        effective_plateau=PaperIEffectivePlateauObservation(
            policy="paper_i_effective_plateau_v1",
            status="available",
            controller_round=28,
            active_ansatz_depth=28,
            absolute_energy_error=errors[27],
            best_observed_error=errors[-1],
            selection_threshold=1.10 * errors[-1],
            available_horizon_controller_rounds=30,
            horizon_scope="deliberately_stopped_prefix",
            algorithmic_work=work,
            prefix={},  # Only serialization shape is under test at this seam.
            resources=None,
            failure=None,
        ),
        append_matched=PaperIAppendMatchedObservation(
            status="unavailable",
            reason="canonical_append_reference_not_found",
            shared_window_end_controller_round=None,
            common_target_absolute_error=None,
            sr_snake=None,
            append_adapt=None,
        ),
        requested_rounds=(),
        canonical_all_work=work,
        horizon_scope="deliberately_stopped_prefix",
        available_controller_rounds=30,
        provenance=PaperIRunProvenance(
            problem_key="hh_l2_test",
            problem_request_sha256="a" * 64,
            problem_family="hh",
            exact_target_label="same_cutoff_ed",
            exact_same_cutoff_energy=exact_energy,
            reference_label="canonical_reference",
            reference_source_label="canonical_reference",
            reference_state_fingerprint="reference-state",
            route_family="ra_adapt",
            route_profile_request="test_route",
            route_profile="test_route",
            route_contract_sha256="b" * 64,
            candidate_representation="macro_generator_v1",
            optimizer="POWELL",
            optimizer_maxiter=200,
            seed=7,
            qiskit_compile_convention="table_i_basis_gate_transpile_v1",
        ),
    )


def test_typed_summary_is_validated_in_the_exact_written_json_shape(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    typed_summary = _typed_round30_summary()
    raw_projection = typed_summary.to_dict()
    assert isinstance(raw_projection["accepted_error_trace"], tuple)

    summary_path = tmp_path / "summary/summary.json"
    written_shape = runner._write_summary_for_validation(
        worker,
        summary_path,
        typed_summary,
    )

    assert isinstance(written_shape["accepted_error_trace"], list)
    assert written_shape["available_controller_rounds"] == 30
    assert summary_path.read_bytes() == (
        worker.canonical_json_bytes(raw_projection) + b"\n"
    )


def test_post_execute_closure_failure_preserves_staging_in_quarantine(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    execution_id = runner.TARGET_EXECUTION_IDS[0]
    staging = tmp_path / "temporary/cell_output"
    (staging / "checkpoints").mkdir(parents=True)
    (staging / "summary").mkdir()
    (staging / "checkpoints/current.json").write_bytes(b'{"round":30}\n')
    (staging / "summary/summary.json").write_bytes(b'{"rounds":30}\n')
    runtime = tmp_path / "runtime"
    (runtime / "quarantine").mkdir(parents=True)

    destination = runner._quarantine_post_execute_failure(
        worker,
        staging=staging,
        runtime_dir=runtime,
        execution_id=execution_id,
        failure=runner.LocalRunError("summary closure drifted"),
    )

    assert destination == runtime / "quarantine" / execution_id
    assert (destination / "checkpoints/current.json").read_bytes() == (
        b'{"round":30}\n'
    )
    assert (destination / "summary/summary.json").read_bytes() == (
        b'{"rounds":30}\n'
    )
    receipt = runner._load_digested(
        worker,
        destination / "quarantine_receipt.json",
        label="test quarantine receipt",
    )
    assert receipt["status"] == "preserved_post_execute_closure_failure"
    assert receipt["execution_id"] == execution_id
    assert receipt["scientific_execution_completed"] is True
    assert receipt["scientific_output_published"] is False
    assert {row["path"] for row in receipt["preserved_artifacts"]} == {
        "checkpoints/current.json",
        "summary/summary.json",
    }


def test_repaired_campaign_uses_new_v2_activation_runtime_and_receipts() -> None:
    runner = _load_runner()

    assert runner.DEFAULT_ACTIVATION_DIR.name.endswith(
        "20260812_v2_local_activation"
    )
    assert runner.DEFAULT_RUNTIME_DIR.name.endswith("20260812_v2")
    assert "20260812_v1" not in runner.DEFAULT_ACTIVATION_DIR.as_posix()
    assert "20260812_v1" not in runner.DEFAULT_RUNTIME_DIR.as_posix()
    for schema in (
        runner.LOCAL_REQUEST_SCHEMA,
        runner.LOCAL_PREFLIGHT_SCHEMA,
        runner.LOCAL_AUTHORIZATION_SCHEMA,
        runner.LOCAL_ACTIVATION_SCHEMA,
        runner.LOCAL_RUNTIME_SCHEMA,
        runner.LOCAL_STATUS_SCHEMA,
        runner.LOCAL_EXECUTION_SCHEMA,
        runner.LOCAL_WORKER_RECEIPT_SCHEMA,
    ):
        assert schema.endswith("_v2")


def test_supervisor_requires_fresh_authenticated_exact_wave_clearance() -> None:
    supervisor = _load_supervisor()
    now = datetime(2026, 8, 13, 3, 0, tzinfo=timezone.utc)
    execution_ids = ("cell-a", "cell-b")
    clearance = {
        "schema": supervisor.REMOTE_CLEARANCE_SCHEMA,
        "status": "passed_authenticated_no_remote_overlap_clearance",
        "wave": 2,
        "execution_ids": list(execution_ids),
        "runner_sha256": "a" * 64,
        "activation_manifest_sha256": "b" * 64,
        "runtime_manifest_sha256": "c" * 64,
        "activation_dir": "/fixed/activation-v2",
        "runtime_dir": "/fixed/runtime-v2",
        "observed_at_utc": (now - timedelta(minutes=2)).isoformat(),
        "valid_until_utc": (now + timedelta(minutes=8)).isoformat(),
        "authentication_kind": (
            "interactive_ssh_duo_condor_q_snapshot_v1"
        ),
        "authenticated_remote_query": True,
        "scheduler": "chtc_condor",
        "scheduler_snapshot_sha256": "d" * 64,
        "remote_active_execution_ids": [],
        "overlapping_execution_ids": [],
        "remote_factories_frozen": True,
        "no_remote_overlap": True,
        "scientific_execution_performed": False,
    }

    observed = supervisor._validate_remote_overlap_clearance(
        clearance,
        wave_number=2,
        execution_ids=execution_ids,
        runner_sha256="a" * 64,
        activation_manifest_sha256="b" * 64,
        runtime_manifest_sha256="c" * 64,
        activation_dir=Path("/fixed/activation-v2"),
        runtime_dir=Path("/fixed/runtime-v2"),
        now=now,
    )
    assert observed["no_remote_overlap"] is True

    overlapping = {**clearance, "remote_active_execution_ids": ["cell-a"]}
    with pytest.raises(
        supervisor.SupervisorError,
        match="remote-overlap clearance drifted",
    ):
        supervisor._validate_remote_overlap_clearance(
            overlapping,
            wave_number=2,
            execution_ids=execution_ids,
            runner_sha256="a" * 64,
            activation_manifest_sha256="b" * 64,
            runtime_manifest_sha256="c" * 64,
            activation_dir=Path("/fixed/activation-v2"),
            runtime_dir=Path("/fixed/runtime-v2"),
            now=now,
        )


def test_supervisor_run_gate_fails_closed_when_wave_clearance_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    supervisor = _load_supervisor()
    activation_dir = tmp_path / "activation-v2"
    runtime_dir = tmp_path / "runtime-v2"
    clearance_dir = tmp_path / "clearances-v2"
    activation_dir.mkdir()
    runtime_dir.mkdir()
    clearance_dir.mkdir()

    def write_digested(path: Path, unsigned: dict) -> dict:
        payload = {
            **unsigned,
            "sha256": supervisor._canonical_sha256(unsigned),
        }
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return payload

    activation = write_digested(
        activation_dir / "activation_manifest.json",
        {
            "schema": runner.LOCAL_ACTIVATION_SCHEMA,
            "waves": [list(wave) for wave in runner.WAVES],
        },
    )
    runtime = write_digested(
        runtime_dir / "runtime_manifest.json",
        {
            "schema": runner.LOCAL_RUNTIME_SCHEMA,
            "activation_manifest_sha256": activation["sha256"],
        },
    )
    monkeypatch.setattr(supervisor, "ACTIVATION_DIR", activation_dir)
    monkeypatch.setattr(supervisor, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(supervisor, "REMOTE_CLEARANCE_DIR", clearance_dir)
    monkeypatch.setattr(
        supervisor,
        "EXPECTED_RUNNER_SHA256",
        supervisor._sha256_file(supervisor.RUNNER),
    )

    with pytest.raises(
        supervisor.SupervisorError,
        match="clearance is absent or unsafe",
    ):
        supervisor._require_remote_overlap_clearance(2)
    assert runtime["activation_manifest_sha256"] == activation["sha256"]
