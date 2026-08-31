"""Focused tests for the authenticated-only CHTC operations adapter."""

from __future__ import annotations

import base64
from dataclasses import replace
import io
import json
import shlex
import hashlib
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.time_dynamics.acquisition_supervisor import (
    SUBMISSION_INTENT_SCHEMA_V1,
    CampaignRegistry,
    CycleGuardFailed,
    LaneHandlers,
    LaneRegistration,
    RemoteAccessPending,
    SchedulerScope,
    SubmissionUnit,
)
from pipelines.time_dynamics.chtc_acquisition_operations import (
    AuthenticatedControlMaster,
    CHTCAcquisitionOperations,
    CHTCOperationsConfig,
    CommandResult,
    build_operations,
)
from pipelines.time_dynamics.guarded_archive_transfer import (
    ArchiveTransferSpec,
    DEFAULT_FREE_FLOOR_BYTES,
    TransferRecoveryError,
    finalize_archive_transfer,
)
from pipelines.time_dynamics.acquisition_registry_contract import (
    PRODUCTION_REQUIRED_BLOCKERS,
)


def _lane(root: Path) -> LaneRegistration:
    return LaneRegistration(
        lane_id="paper",
        kind="paper_ii",
        monitor_enabled=True,
        submission_enabled=True,
        attempts_root=root / "attempts",
        state_root=root / "lane_state",
        scientific_state=root / "ledger.json",
        report_contract=root / "report.json",
        planner_authority="planner.next_actions",
    )


def _registry(root: Path) -> CampaignRegistry:
    lane = _lane(root)
    assert lane.scientific_state is not None
    assert lane.report_contract is not None
    lane.scientific_state.write_text("{}\n")
    lane.report_contract.write_text("{}\n")
    return CampaignRegistry(
        registry_id="test_registry",
        repository_root=root,
        supervisor_state_root=root / "supervisor",
        lanes=(lane,),
    )


class FakeRunner:
    def __init__(
        self,
        *,
        queue_rows: list[dict],
        history_rows: list[dict] | None = None,
        check_returncode: int = 0,
        submitted_cluster: int = 99,
        remote_files: dict[str, bytes] | None = None,
        lose_after_check: bool = False,
    ) -> None:
        self.queue_rows = queue_rows
        self.history_rows = history_rows or []
        self.check_returncode = check_returncode
        self.submitted_cluster = submitted_cluster
        self.remote_files = dict(remote_files or {})
        self.lose_after_check = lose_after_check
        self.argvs: list[tuple[str, ...]] = []

    def __call__(
        self, argv: tuple[str, ...], *, timeout_seconds: int
    ) -> CommandResult:
        assert timeout_seconds > 0
        self.argvs.append(argv)
        if "-O" in argv:
            return CommandResult(
                returncode=self.check_returncode,
                stderr="no authenticated master"
                if self.check_returncode
                else "",
            )
        if self.lose_after_check:
            return CommandResult(returncode=255, stderr="mux socket disappeared")
        remote = shlex.split(argv[-1])
        source_index = remote.index("-c") + 1
        source = remote[source_index]
        args = remote[source_index + 1 :]
        if "base64.b64encode" in source:
            payload = self.remote_files[args[0]]
            offset = int(args[1])
            length = int(args[2])
            chunk = payload[offset : offset + length]
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "path": args[0],
                        "offset": offset,
                        "length": length,
                        "sha256": hashlib.sha256(chunk).hexdigest(),
                        "base64": base64.b64encode(chunk).decode("ascii"),
                    }
                ),
            )
        if "remaining = length" in source:
            payload = self.remote_files[args[0]]
            length = int(args[1])
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "path": args[0],
                        "length": length,
                        "sha256": hashlib.sha256(payload[:length]).hexdigest(),
                    }
                ),
            )
        if "member_count" in source and "gzip_integrity" in source:
            payload = self.remote_files[args[0]]
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                member_count = sum(1 for _ in archive)
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "path": args[0],
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "member_count": member_count,
                        "gzip_integrity": True,
                        "tar_integrity": True,
                    }
                ),
            )
        if "chtc_combined_condor_capture_v1" in source:
            scope_rows = json.loads(args[1])
            intents = json.loads(args[2])
            envelope = {
                "schema": "chtc_combined_condor_capture_v1",
                "captured_at_unix_ns": 1787928000123456000,
                "scope_rows": scope_rows,
                "intent_rows": intents,
                "queue": {
                    "returncode": 0,
                    "stdout": json.dumps(self.queue_rows),
                    "stderr": "",
                },
                "history": {
                    "returncode": 0,
                    "stdout": json.dumps(self.history_rows),
                    "stderr": "",
                },
            }
            return CommandResult(returncode=0, stdout=json.dumps(envelope))
        assert "validate_condor_submit_lifecycle.py" in source
        expected_count = int(args[5])
        return CommandResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "submitted",
                    "cluster_id": self.submitted_cluster,
                    "process_ids": list(range(expected_count)),
                    "job_batch_name": args[4],
                    "unit_id": args[2],
                    "initial_queue_count": expected_count,
                }
            ),
        )


def _adapter(
    root: Path,
    runner: FakeRunner,
    *,
    handlers: dict[str, LaneHandlers] | None = None,
    archive_transfers: dict[
        tuple[str, int, tuple[int, ...]], ArchiveTransferSpec
    ]
    | None = None,
) -> CHTCAcquisitionOperations:
    transport = AuthenticatedControlMaster(
        socket_path=root / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=runner,
    )
    config = CHTCOperationsConfig(
        repository_root=root,
        transport=transport,
        bootstrap_action_ids={("paper", 42, 0): "action-0"},
        remote_workdirs={"paper": "/home/jsstrobel/fixed-paper-workdir"},
        archive_transfers=archive_transfers or {},
        history_since_epoch=1787800000,
    )
    return CHTCAcquisitionOperations(
        config, handlers or {"paper": LaneHandlers()}
    )


def _archive_payload() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        content = b"binary-safe\x00archive\xffpayload"
        info = tarfile.TarInfo("run.json")
        info.size = len(content)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _archive_spec(root: Path, payload: bytes) -> ArchiveTransferSpec:
    return ArchiveTransferSpec(
        remote_path="/home/jsstrobel/exact/archive.tar.gz",
        local_final_path=root / "output/evidence/archive.tar.gz",
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_count=1,
    )


def _active_row() -> dict:
    return {
        "ClusterId": 42,
        "ProcId": 0,
        "JobStatus": 2,
        "JobBatchName": "existing-batch",
    }


def _write_intent(
    registry: CampaignRegistry, unit: SubmissionUnit
) -> None:
    lane = registry.lane("paper")
    path = lane.state_root / "submission_intents" / f"{unit.unit_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": SUBMISSION_INTENT_SCHEMA_V1,
                "lane_id": lane.lane_id,
                "unit": unit.to_json_dict(),
            }
        )
        + "\n"
    )


def _submission_unit() -> SubmissionUnit:
    return SubmissionUnit(
        unit_id="frontier-1",
        action_ids=("next-a", "next-b"),
        payload={
            "job_batch_name": "paper-frontier-1",
            "submit_descriptor": "chtc/package/submit.sub",
            "expected_process_count": 2,
            "submission_scope": {
                "process_ids": [0, 1],
                "action_ids": ["next-a", "next-b"],
            },
        },
    )


def test_capture_uses_only_existing_batch_mode_master_and_binds_raw_evidence(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner(queue_rows=[_active_row()])
    adapter = _adapter(tmp_path, runner)
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))

    snapshot = adapter.capture_snapshot(registry, expected_scopes=(scope,))

    assert snapshot.jobs[0].state == "running"
    assert snapshot.jobs[0].action_id == "action-0"
    evidence = snapshot.source["raw_scheduler_evidence"]
    for query in ("queue", "history"):
        path = tmp_path / evidence[query]["reference"]
        assert path.is_file()
    assert len(runner.argvs) == 2
    assert runner.argvs[0][runner.argvs[0].index("-O") + 1] == "check"
    assert "BatchMode=yes" in runner.argvs[0]
    assert "BatchMode=yes" in runner.argvs[1]
    assert runner.argvs[1][:3] == ("/usr/bin/ssh", "-F", "/dev/null")
    assert "ProxyCommand=/usr/bin/false" in runner.argvs[1]
    assert "PasswordAuthentication=no" in runner.argvs[1]
    assert "PubkeyAuthentication=no" in runner.argvs[1]
    assert not any("duo" in value.lower() for argv in runner.argvs for value in argv)


def test_missing_authenticated_master_fails_closed_and_writes_resumable_state(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner(queue_rows=[_active_row()], check_returncode=255)
    adapter = _adapter(tmp_path, runner)
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))

    with pytest.raises(RemoteAccessPending, match="remote_access_pending"):
        adapter.capture_snapshot(registry, expected_scopes=(scope,))

    state = json.loads(
        (
            tmp_path
            / "output/time_dynamics_acquisition/production_transport_state.json"
        ).read_text()
    )
    assert state["status"] == "remote_access_pending"
    assert len(runner.argvs) == 1


def test_socket_loss_after_precheck_cannot_fall_back_to_new_ssh(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(queue_rows=[], lose_after_check=True)
    transport = AuthenticatedControlMaster(
        socket_path=tmp_path / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=runner,
    )

    with pytest.raises(RemoteAccessPending, match="mux socket disappeared"):
        transport.archive_identity("/home/jsstrobel/exact/archive.tar.gz")

    assert len(runner.argvs) == 2
    command = runner.argvs[1]
    assert command[:3] == ("/usr/bin/ssh", "-F", "/dev/null")
    assert f"ControlPath={tmp_path / 'fixed.sock'}" in command
    assert "ProxyCommand=/usr/bin/false" in command
    assert "PasswordAuthentication=no" in command
    assert "KbdInteractiveAuthentication=no" in command
    assert "PubkeyAuthentication=no" in command


def test_transport_timeout_is_typed_as_remote_access_pending(
    tmp_path: Path,
) -> None:
    def timeout_runner(argv, *, timeout_seconds):
        del timeout_seconds
        if "-O" in argv:
            return CommandResult(returncode=0)
        raise subprocess.TimeoutExpired(argv, 5)

    transport = AuthenticatedControlMaster(
        socket_path=tmp_path / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=timeout_runner,
    )
    with pytest.raises(RemoteAccessPending):
        transport.archive_identity("/home/jsstrobel/exact/archive.tar.gz")


def test_non_utf8_remote_output_is_typed_and_records_transport_state(
    tmp_path: Path,
) -> None:
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)

    def invalid_utf8_runner(argv, *, timeout_seconds):
        del timeout_seconds
        if "-O" in argv:
            return CommandResult(returncode=0)
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    transport = AuthenticatedControlMaster(
        socket_path=tmp_path / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=invalid_utf8_runner,
    )
    adapter = CHTCAcquisitionOperations(
        CHTCOperationsConfig(
            repository_root=tmp_path,
            transport=transport,
            archive_transfers={("paper", 42, (0,)): spec},
        ),
        {"paper": LaneHandlers()},
    )

    with pytest.raises(RemoteAccessPending, match="non-UTF-8"):
        adapter.fetch_verified_archive("paper", 42, (0,))

    state = json.loads(
        (
            tmp_path
            / "output/time_dynamics_acquisition/production_transport_state.json"
        ).read_text()
    )
    assert state["status"] == "remote_access_pending"
    assert state["phase"] == "archive_transfer"
    assert not spec.part_path.exists()
    assert not spec.local_final_path.exists()


def test_malformed_remote_numeric_fields_fail_as_typed_cycle_guards(
    tmp_path: Path,
) -> None:
    def malformed_runner(argv, *, timeout_seconds):
        del timeout_seconds
        if "-O" in argv:
            return CommandResult(returncode=0)
        remote = shlex.split(argv[-1])
        source = remote[remote.index("-c") + 1]
        if "remaining = length" in source:
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "path": "/home/jsstrobel/exact/archive.tar.gz",
                        "length": "not-an-integer",
                        "sha256": "0" * 64,
                    }
                ),
            )
        return CommandResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "path": "/home/jsstrobel/exact/archive.tar.gz",
                    "offset": "not-an-integer",
                    "length": 1,
                    "sha256": "0" * 64,
                    "base64": "AA==",
                }
            ),
        )

    transport = AuthenticatedControlMaster(
        socket_path=tmp_path / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=malformed_runner,
    )
    with pytest.raises(CycleGuardFailed, match="archive_prefix_unverifiable"):
        transport.archive_prefix_sha256(
            "/home/jsstrobel/exact/archive.tar.gz", 1
        )
    with pytest.raises(CycleGuardFailed, match="archive_chunk_unverifiable"):
        transport.archive_chunk("/home/jsstrobel/exact/archive.tar.gz", 0, 1)


def test_new_unit_is_indeterminate_until_a_later_snapshot_queries_its_intent(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner(queue_rows=[_active_row()])
    adapter = _adapter(tmp_path, runner)
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))
    unit = _submission_unit()

    snapshot = adapter.capture_snapshot(registry, expected_scopes=(scope,))
    handlers = adapter.handlers_for(registry)["paper"]
    assert handlers.reconcile_submission is not None
    assert handlers.submit_unit is not None
    first = handlers.reconcile_submission(registry.lane("paper"), unit, snapshot)
    assert first.status == "indeterminate"
    with pytest.raises(CycleGuardFailed, match="submission_scope_not_verified"):
        handlers.submit_unit(registry.lane("paper"), unit)

    _write_intent(registry, unit)
    snapshot = adapter.capture_snapshot(registry, expected_scopes=(scope,))
    second = handlers.reconcile_submission(registry.lane("paper"), unit, snapshot)
    assert second.status == "verified_absent"
    remote = handlers.submit_unit(registry.lane("paper"), unit)
    assert remote["cluster_id"] == 99
    assert remote["process_ids"] == [0, 1]
    assert len(runner.argvs) == 6


def test_intent_query_recovers_existing_submission_without_resubmitting(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    unit = _submission_unit()
    _write_intent(registry, unit)
    runner = FakeRunner(
        queue_rows=[
            _active_row(),
            {
                "ClusterId": 88,
                "ProcId": 0,
                "JobStatus": 1,
                "JobBatchName": "paper-frontier-1",
                "AcquisitionUnitId": "frontier-1",
            },
            {
                "ClusterId": 88,
                "ProcId": 1,
                "JobStatus": 2,
                "JobBatchName": "paper-frontier-1",
                "AcquisitionUnitId": "frontier-1",
            },
        ]
    )
    adapter = _adapter(tmp_path, runner)
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))
    snapshot = adapter.capture_snapshot(registry, expected_scopes=(scope,))
    handlers = adapter.handlers_for(registry)["paper"]
    assert handlers.reconcile_submission is not None
    result = handlers.reconcile_submission(registry.lane("paper"), unit, snapshot)

    assert result.status == "found"
    assert result.remote["cluster_id"] == 88
    assert len(runner.argvs) == 2


def test_capture_rejects_missing_action_identity_binding(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner(
        queue_rows=[
            {
                "ClusterId": 42,
                "ProcId": 0,
                "JobStatus": 2,
            }
        ]
    )
    transport = AuthenticatedControlMaster(
        socket_path=tmp_path / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=runner,
    )
    adapter = CHTCAcquisitionOperations(
        CHTCOperationsConfig(
            repository_root=tmp_path,
            transport=transport,
        ),
        {"paper": LaneHandlers()},
    )
    scope = SchedulerScope(lane_id="paper", cluster_id=42, process_ids=(0,))

    with pytest.raises(CycleGuardFailed, match="scheduler_action_binding_missing"):
        adapter.capture_snapshot(registry, expected_scopes=(scope,))


def test_archive_transport_fetches_binary_bytes_through_exact_registered_part(
    tmp_path: Path,
) -> None:
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    runner = FakeRunner(
        queue_rows=[], remote_files={spec.remote_path: payload}
    )
    adapter = _adapter(
        tmp_path,
        runner,
        archive_transfers={("paper", 42, (0,)): spec},
    )

    receipt = adapter.fetch_verified_archive("paper", 42, (0,), chunk_bytes=127)

    assert spec.local_final_path.read_bytes() == payload
    assert not spec.part_path.exists()
    assert receipt["archive_sha256"] == spec.expected_sha256
    assert receipt["tar_member_count"] == 1
    assert receipt["remote_preserved"] is True
    assert runner.remote_files[spec.remote_path] == payload
    remote_commands = [argv[-1] for argv in runner.argvs if "-O" not in argv]
    assert remote_commands
    assert all("os.unlink" not in command for command in remote_commands)
    assert all("os.remove" not in command for command in remote_commands)
    assert all("BatchMode=yes" in argv for argv in runner.argvs)


def test_archive_transport_resumes_only_after_exact_remote_prefix_proof(
    tmp_path: Path,
) -> None:
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    spec.part_path.parent.mkdir(parents=True)
    split = len(payload) // 3
    spec.part_path.write_bytes(payload[:split])
    runner = FakeRunner(
        queue_rows=[], remote_files={spec.remote_path: payload}
    )
    adapter = _adapter(
        tmp_path,
        runner,
        archive_transfers={("paper", 42, (0,)): spec},
    )

    adapter.fetch_verified_archive("paper", 42, (0,), chunk_bytes=113)

    assert spec.local_final_path.read_bytes() == payload
    recovery = json.loads(spec.effective_recovery_receipt_path.read_text())
    assert recovery["status"] == "prefix_verified"
    assert recovery["prefix_proof"]["length"] == split
    assert recovery["prefix_proof"]["matches"] is True


def test_archive_transport_identity_mismatch_blocks_before_local_part(
    tmp_path: Path,
) -> None:
    payload = _archive_payload()
    bound = _archive_spec(tmp_path, payload)
    spec = ArchiveTransferSpec(
        remote_path=bound.remote_path,
        local_final_path=bound.local_final_path,
        expected_bytes=bound.expected_bytes,
        expected_sha256="0" * 64,
        expected_member_count=bound.expected_member_count,
    )
    runner = FakeRunner(
        queue_rows=[], remote_files={spec.remote_path: payload}
    )
    adapter = _adapter(
        tmp_path,
        runner,
        archive_transfers={("paper", 42, (0,)): spec},
    )

    with pytest.raises(CycleGuardFailed, match="remote_archive_binding_mismatch"):
        adapter.fetch_verified_archive("paper", 42, (0,))

    assert not spec.part_path.exists()
    assert not spec.local_final_path.exists()
    state = json.loads(
        (
            tmp_path
            / "output/time_dynamics_acquisition/production_transport_state.json"
        ).read_text()
    )
    assert state["status"] == "archive_transfer_blocked"


def test_archive_transport_unavailable_master_never_starts_transfer(
    tmp_path: Path,
) -> None:
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    runner = FakeRunner(
        queue_rows=[],
        remote_files={spec.remote_path: payload},
        check_returncode=255,
    )
    adapter = _adapter(
        tmp_path,
        runner,
        archive_transfers={("paper", 42, (0,)): spec},
    )

    with pytest.raises(RemoteAccessPending):
        adapter.fetch_verified_archive("paper", 42, (0,))

    assert len(runner.argvs) == 1
    assert not spec.part_path.exists()
    assert not spec.local_final_path.exists()


def test_archive_fetch_rejects_unregistered_caller_identity(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(queue_rows=[])
    adapter = _adapter(tmp_path, runner)
    with pytest.raises(CycleGuardFailed, match="archive_transfer_binding_missing"):
        adapter.fetch_verified_archive("paper", 42, (0,))
    assert runner.argvs == []


def test_ingestion_handler_requires_and_consumes_exact_archive_binding(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    runner = FakeRunner(
        queue_rows=[], remote_files={spec.remote_path: payload}
    )
    observed: list[tuple[int, tuple[int, ...]]] = []

    def ingest_delta(
        lane, cluster_id, process_ids, selected_jobs, *, verified_archive
    ):
        del lane, selected_jobs
        observed.append((cluster_id, process_ids))
        consumed_path = tmp_path / verified_archive["final_path"]
        assert hashlib.sha256(consumed_path.read_bytes()).hexdigest() == (
            verified_archive["archive_sha256"]
        )
        with tarfile.open(consumed_path, "r:gz") as archive:
            assert sum(1 for _ in archive) == verified_archive[
                "archive_member_count"
            ]
        return {
            "status": "ingested",
            "consumed_verified_archive_sha256": verified_archive[
                "verified_archive_binding_sha256"
            ],
            "consumed_verified_archive": {
                "verified_archive_binding_sha256": verified_archive[
                    "verified_archive_binding_sha256"
                ],
                "final_path": verified_archive["final_path"],
                "archive_sha256": verified_archive["archive_sha256"],
                "finalization_receipt_sha256": verified_archive[
                    "finalization_receipt"
                ]["sha256"],
            },
        }

    adapter = _adapter(
        tmp_path,
        runner,
        handlers={"paper": LaneHandlers(ingest_delta=ingest_delta)},
        archive_transfers={("paper", 42, (0,)): spec},
    )
    handler = adapter.handlers_for(registry)["paper"].ingest_delta
    assert handler is not None

    outcome = handler(registry.lane("paper"), 42, (0,), ())

    assert observed == [(42, (0,))]
    assert outcome["status"] == "ingested"
    assert outcome["verified_archive_transfer"]["remote_preserved"] is True


def test_recovered_archive_rechecks_storage_floor_before_ingestion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    payload = _archive_payload()
    spec = replace(
        _archive_spec(tmp_path, payload), extraction_reserve_bytes=1
    )
    spec.part_path.parent.mkdir(parents=True)
    spec.part_path.write_bytes(payload)
    finalize_archive_transfer(spec)
    runner = FakeRunner(queue_rows=[], remote_files={spec.remote_path: payload})
    called: list[bool] = []

    def ingest_delta(*_args, **_kwargs):
        called.append(True)
        return {"status": "unexpected"}

    adapter = _adapter(
        tmp_path,
        runner,
        handlers={"paper": LaneHandlers(ingest_delta=ingest_delta)},
        archive_transfers={("paper", 42, (0,)): spec},
    )
    monkeypatch.setattr(
        "pipelines.time_dynamics.guarded_archive_transfer.shutil.disk_usage",
        lambda _path: SimpleNamespace(
            total=DEFAULT_FREE_FLOOR_BYTES,
            used=0,
            free=DEFAULT_FREE_FLOOR_BYTES,
        ),
    )
    handler = adapter.handlers_for(registry)["paper"].ingest_delta
    assert handler is not None

    with pytest.raises(
        CycleGuardFailed, match="archive_ingestion_storage_guard_failed"
    ):
        handler(registry.lane("paper"), 42, (0,), ())

    assert called == []


def test_ingestion_rereads_and_rejects_tampered_finalization_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    runner = FakeRunner(queue_rows=[], remote_files={spec.remote_path: payload})
    called = False

    def ingest_delta(
        _lane, _cluster_id, _process_ids, _selected_jobs, *, verified_archive
    ):
        nonlocal called
        called = True
        return {
            "status": "ingested",
            "consumed_verified_archive_sha256": verified_archive[
                "verified_archive_binding_sha256"
            ],
            "consumed_verified_archive": {
                "verified_archive_binding_sha256": verified_archive[
                    "verified_archive_binding_sha256"
                ],
                "final_path": verified_archive["final_path"],
                "archive_sha256": verified_archive["archive_sha256"],
                "finalization_receipt_sha256": verified_archive[
                    "finalization_receipt"
                ]["sha256"],
            },
        }

    adapter = _adapter(
        tmp_path,
        runner,
        handlers={"paper": LaneHandlers(ingest_delta=ingest_delta)},
        archive_transfers={("paper", 42, (0,)): spec},
    )
    original_fetch = adapter.fetch_verified_archive

    def fetch_then_tamper(lane_id, cluster_id, process_ids):
        result = original_fetch(lane_id, cluster_id, process_ids)
        receipt = json.loads(spec.effective_finalization_receipt_path.read_text())
        receipt["archive_bytes"] += 1
        spec.effective_finalization_receipt_path.write_text(
            json.dumps(receipt) + "\n"
        )
        return result

    monkeypatch.setattr(adapter, "fetch_verified_archive", fetch_then_tamper)
    handler = adapter.handlers_for(registry)["paper"].ingest_delta
    assert handler is not None
    with pytest.raises(
        CycleGuardFailed, match="archive_finalization_receipt_unverifiable"
    ):
        handler(registry.lane("paper"), 42, (0,), ())
    assert called is False


def test_ingestion_rejects_stale_or_alternate_archive_consumption(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    runner = FakeRunner(queue_rows=[], remote_files={spec.remote_path: payload})

    def ingest_delta(
        _lane, _cluster_id, _process_ids, _selected_jobs, *, verified_archive
    ):
        return {
            "status": "ingested",
            "consumed_verified_archive_sha256": verified_archive[
                "verified_archive_binding_sha256"
            ],
            "consumed_verified_archive": {
                "verified_archive_binding_sha256": verified_archive[
                    "verified_archive_binding_sha256"
                ],
                "final_path": "output/evidence/alternate.tar.gz",
                "archive_sha256": verified_archive["archive_sha256"],
                "finalization_receipt_sha256": verified_archive[
                    "finalization_receipt"
                ]["sha256"],
            },
        }

    adapter = _adapter(
        tmp_path,
        runner,
        handlers={"paper": LaneHandlers(ingest_delta=ingest_delta)},
        archive_transfers={("paper", 42, (0,)): spec},
    )
    handler = adapter.handlers_for(registry)["paper"].ingest_delta
    assert handler is not None

    with pytest.raises(
        CycleGuardFailed,
        match="ingestion_verified_archive_consumption_unverifiable",
    ):
        handler(registry.lane("paper"), 42, (0,), ())


def test_ingestion_handler_fails_closed_without_exact_archive_binding(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    runner = FakeRunner(queue_rows=[])
    adapter = _adapter(
        tmp_path,
        runner,
        handlers={
            "paper": LaneHandlers(
                ingest_delta=lambda *_args: {"status": "must-not-run"}
            )
        },
    )
    handler = adapter.handlers_for(registry)["paper"].ingest_delta
    assert handler is not None

    with pytest.raises(CycleGuardFailed, match="archive_transfer_binding_missing"):
        handler(registry.lane("paper"), 42, (0,), ())

    assert runner.argvs == []


def test_archive_transfer_paths_are_confined_and_may_not_traverse_symlinks(
    tmp_path: Path,
) -> None:
    payload = _archive_payload()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    escaped = ArchiveTransferSpec(
        remote_path="/home/jsstrobel/exact/archive.tar.gz",
        local_final_path=outside / "archive.tar.gz",
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_count=1,
    )
    transport = AuthenticatedControlMaster(
        socket_path=tmp_path / "fixed.sock",
        endpoint="jsstrobel@ap2001.chtc.wisc.edu",
        runner=FakeRunner(queue_rows=[]),
    )
    with pytest.raises(ValueError, match="inside repository_root"):
        CHTCOperationsConfig(
            repository_root=tmp_path,
            transport=transport,
            archive_transfers={("paper", 42, (0,)): escaped},
        )

    symlink_parent = tmp_path / "output" / "linked"
    symlink_parent.parent.mkdir(parents=True)
    symlink_parent.symlink_to(tmp_path / "real", target_is_directory=True)
    linked = ArchiveTransferSpec(
        remote_path="/home/jsstrobel/exact/archive.tar.gz",
        local_final_path=symlink_parent / "archive.tar.gz",
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_member_count=1,
    )
    with pytest.raises(ValueError, match="symlink"):
        CHTCOperationsConfig(
            repository_root=tmp_path,
            transport=transport,
            archive_transfers={("paper", 42, (0,)): linked},
        )


def test_archive_transfer_rejects_parent_symlink_swap_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _archive_payload()
    spec = _archive_spec(tmp_path, payload)
    spec.local_final_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    runner = FakeRunner(queue_rows=[], remote_files={spec.remote_path: payload})
    adapter = _adapter(
        tmp_path,
        runner,
        archive_transfers={("paper", 42, (0,)): spec},
    )
    original = CHTCOperationsConfig._validate_archive_spec_paths
    swapped = False

    def validate_then_swap(config, candidate):
        nonlocal swapped
        original(config, candidate)
        if not swapped:
            swapped = True
            holding = spec.local_final_path.parent.with_name("evidence-pinned")
            spec.local_final_path.parent.rename(holding)
            spec.local_final_path.parent.symlink_to(
                outside, target_is_directory=True
            )

    monkeypatch.setattr(
        CHTCOperationsConfig, "_validate_archive_spec_paths", validate_then_swap
    )

    with pytest.raises(
        (CycleGuardFailed, TransferRecoveryError),
        match="path|symlink|nofollow|confinement",
    ):
        adapter.fetch_verified_archive("paper", 42, (0,), chunk_bytes=127)

    assert tuple(outside.iterdir()) == ()


def test_build_operations_invokes_activation_validation_before_failing_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path)
    observed: list[str] = []

    def reject(_registry):
        observed.append(_registry.registry_id)
        raise RuntimeError("activation validator called")

    monkeypatch.setattr(
        "pipelines.time_dynamics.acquisition_registry_contract."
        "validate_registry_for_activation",
        reject,
    )
    with pytest.raises(RuntimeError, match="activation validator called"):
        build_operations(registry)
    assert observed == ["test_registry"]


def test_canonical_registry_exposes_adapter_as_machine_readable_block() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = CampaignRegistry.read(
        root
        / "pipelines/time_dynamics/campaigns/"
        "time_dynamics_acquisition_registry_v1.json",
        repository_root=root,
    )

    assert registry.production_operations is None
    blocked = registry.production_operations_blocked
    assert blocked is not None
    assert blocked.adapter_id == "chtc_authenticated_controlmaster_v1"
    assert set(PRODUCTION_REQUIRED_BLOCKERS).issubset(blocked.blocked_by)
    assert blocked.source_path.is_file()
