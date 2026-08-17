from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
)
SUPERVISOR_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "supervise_local_page16_insertion_comparator_k50_continuations_"
    "20260813.py"
)
PAGE12_RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page12_insertion_comparators_20260812.py"
)


@pytest.fixture(autouse=True)
def _restore_process_import_state_after_test():
    """Keep sealed-source activation from contaminating later test modules."""

    def is_sealed_namespace(name: str) -> bool:
        return (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
            or name == "package_contract"
            or name.startswith("paper_i_page16_")
        )

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if is_sealed_namespace(name)
    }
    saved_path = list(sys.path)
    saved_importer_cache = dict(sys.path_importer_cache)
    saved_environ = dict(os.environ)
    saved_dont_write_bytecode = sys.dont_write_bytecode
    saved_cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        for name in tuple(sys.modules):
            if is_sealed_namespace(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(saved_importer_cache)
        os.environ.clear()
        os.environ.update(saved_environ)
        sys.dont_write_bytecode = saved_dont_write_bytecode
        importlib.invalidate_caches()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _adapter():
    return _load(
        ADAPTER_PATH,
        "paper_i_page16_local_insertion_comparator_k50_adapter_test",
    )


def _supervisor():
    return _load(
        SUPERVISOR_PATH,
        "paper_i_page16_local_insertion_comparator_k50_supervisor_test",
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_campaign_has_three_terminal_k50_cells_and_nine_conditional_k30_cells() -> None:
    adapter = _adapter()

    assert len(adapter.TERMINAL_CHTC_EXECUTION_IDS) == 3
    assert len(adapter.CONDITIONAL_EXECUTION_IDS) == 9
    assert adapter.SW_ALWAYS_CHTC_EXECUTION_ID in (
        adapter.TERMINAL_CHTC_EXECUTION_IDS
    )
    assert adapter.SW_ALWAYS_CHTC_EXECUTION_ID not in (
        adapter.CONDITIONAL_EXECUTION_IDS
    )
    assert not (
        set(adapter.TERMINAL_CHTC_EXECUTION_IDS)
        & set(adapter.CONDITIONAL_EXECUTION_IDS)
    )
    assert set(adapter.TERMINAL_CHTC_EXECUTION_IDS) | set(
        adapter.CONDITIONAL_EXECUTION_IDS
    ) == set(adapter.k30.PACKAGE_EXECUTION_IDS)
    assert adapter.MAX_CONCURRENCY == 1
    assert adapter.SOURCE_HORIZON == 30
    assert adapter.TARGET_HORIZON == 50


def test_terminal_status_waits_for_exact_sw_closure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "SW_ALWAYS_CLOSURE_RECEIPT_PATH",
        tmp_path / "absent.json",
    )
    monkeypatch.setattr(
        adapter,
        "_authenticate_terminal_chtc_archive",
        lambda _worker, *, execution_id, job: adapter._digested(
            {
                "schema": adapter.TERMINAL_CHTC_SCHEMA,
                "status": "passed_authenticated_k50_terminal_exclusion",
                "execution_id": execution_id,
            }
        ),
    )

    status = adapter.terminal_chtc_status()

    assert status["status"] == (
        "waiting_for_authenticated_sw_always_closure_and_remote_"
        "materialization_exclusion"
    )
    assert status["all_terminal_cells_authenticated"] is False
    assert status["authenticated_terminal_count"] == 2
    assert status["pending_execution_ids"] == [
        adapter.SW_ALWAYS_CHTC_EXECUTION_ID
    ]
    assert status["scientific_execution_performed"] is False


def test_prepare_cannot_materialize_before_sw_materialization_exclusion_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "SW_ALWAYS_CLOSURE_RECEIPT_PATH",
        tmp_path / "absent.json",
    )
    activation = tmp_path / "activation"

    with pytest.raises(adapter.ContinuationError, match="Waiting for authenticated"):
        adapter.prepare_activation(activation_dir=activation)

    assert not activation.exists()


def test_sw_closure_receipt_authenticates_archive_and_remote_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    worker = adapter.k30._load_worker()
    execution_id = adapter.SW_ALWAYS_CHTC_EXECUTION_ID
    job = adapter._job_by_id(worker)[execution_id]
    manifest_path = f"runs/{execution_id}/execution_manifest.json"
    manifest = worker.digested(
        {
            "schema": "paper_i_ra_adapt_page16_macro_phase23_qiskit_execution_manifest_v1",
            "status": "passed",
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "comparator_policy": job["comparator_policy"],
            "target_horizon": 50,
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
        }
    )
    worker_receipt = worker.digested(
        {
            "schema": "paper_i_ra_adapt_page16_macro_phase23_qiskit_worker_receipt_v1",
            "status": "passed",
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "execution_manifest_sha256": manifest["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
        }
    )
    archive = tmp_path / "archive.tar.gz"
    manifest_file = tmp_path / "execution_manifest.json"
    worker_file = tmp_path / "worker_receipt.json"
    _write_json(manifest_file, manifest)
    _write_json(worker_file, worker_receipt)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(worker_file, arcname="./worker_receipt.json")
        stream.add(manifest_file, arcname=f"./{manifest_path}")
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_size = archive.stat().st_size
    monkeypatch.setattr(adapter, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        adapter,
        "SW_ALWAYS_LOCAL_ARCHIVE_RELATIVE_PATH",
        "archive.tar.gz",
    )
    receipt_path = tmp_path / "closure.json"
    monkeypatch.setattr(adapter, "SW_ALWAYS_CLOSURE_RECEIPT_PATH", receipt_path)
    receipt = worker.digested(
        {
            "schema": (
                "paper_i_ra_adapt_page16_sw_always_"
                "remote_materialization_exclusion_receipt_v2"
            ),
            "status": (
                "passed_sw_always_k50_closed_remote_materialization_excluded"
            ),
            "completed_remote_cell": {
                "regime_id": "strong_weak_u8",
                "comparator_policy": "always_commutation_reduced",
                "typed_insertion_kind": job["typed_insertion_kind"],
                "runtime_insertion_mode": job["runtime_insertion_mode"],
                "execution_id": execution_id,
                "cluster_id": 9_647_386,
                "proc_id": 1,
                "controller_rounds_completed": 50,
                "archive": {
                    "path": "archive.tar.gz",
                    "remote_path": adapter.SW_ALWAYS_REMOTE_ARCHIVE_PATH,
                    "remote_size_bytes": archive_size,
                    "local_size_bytes": archive_size,
                    "size_bytes": archive_size,
                    "remote_sha256": archive_sha256,
                    "local_sha256": archive_sha256,
                    "sha256": archive_sha256,
                },
                "worker_receipt": {
                    "path_inside_archive": "worker_receipt.json",
                    "canonical_sha256": worker_receipt["sha256"],
                    "schema": worker_receipt["schema"],
                    "status": "passed",
                },
                "execution_manifest": {
                    "path_inside_archive": manifest_path,
                    "canonical_sha256": manifest["sha256"],
                },
                "history": {
                    "cluster_id": 9_647_386,
                    "proc_id": 1,
                    "job_status": 4,
                    "exit_code": 0,
                    "num_job_starts": 1,
                    "completion_date_epoch": 1_786_500_000,
                },
                "authenticated_full_sealed_closure": True,
            },
            "remote_materialization_exclusion": {
                "outcome": (
                    "factory_retained_paused_at_completed_prefix_"
                    "after_acknowledged_removal"
                ),
                "removal_command": "condor_rm 9647386",
                "removal_attempts_authenticated": True,
                "before_snapshot": {
                    "job_materialize_paused": 1,
                    "job_materialize_next_proc_id": 2,
                    "materialized_proc_ids": [],
                    "history_completed_proc_ids": [0, 1],
                },
                "after_snapshot": {
                    "cluster_present_in_queue": False,
                    "factory_present": True,
                    "factory_materialization_paused": True,
                    "job_materialize_limit": 2,
                    "job_materialize_max_idle": 0,
                    "job_materialize_next_proc_id": 2,
                    "history_completed_proc_ids": [0, 1],
                },
                "latent_proc_ids_never_materialized": list(range(2, 11)),
                "queue_cluster_absent": True,
                "remote_materialization_excluded": True,
            },
            "authentication": {
                "authenticated_remote_query": True,
                "kind": "interactive_ssh_duo_condor_q_snapshot_v1",
                "source_host": "ap2001.chtc.wisc.edu",
            },
            "scientific_execution_performed_by_action": False,
        }
    )
    _write_json(receipt_path, receipt)

    authenticated = adapter._authenticate_sw_always_closure(
        worker,
        job=job,
    )

    assert authenticated["execution_id"] == execution_id
    assert authenticated["controller_rounds_completed"] == 50
    assert authenticated["source_closure_receipt_sha256"] == receipt["sha256"]
    assert authenticated["remote_materialization_exclusion_authenticated"] is True

    invalid = {
        key: value for key, value in receipt.items() if key != "sha256"
    }
    invalid["remote_materialization_exclusion"] = {
        **invalid["remote_materialization_exclusion"],
        "remote_materialization_excluded": False,
    }
    _write_json(receipt_path, worker.digested(invalid))
    with pytest.raises(
        adapter.ContinuationError,
        match="remote materialization exclusion",
    ):
        adapter._authenticate_sw_always_closure(worker, job=job)

    v1 = worker.digested(
        {
            **{
                key: value
                for key, value in receipt.items()
                if key not in {"sha256", "remote_materialization_exclusion"}
            },
            "schema": (
                "paper_i_ra_adapt_page16_sw_always_closure_"
                "factory_retirement_receipt_v1"
            ),
            "status": (
                "passed_sw_always_k50_closed_factory_retired_without_"
                "latent_materialization"
            ),
            "factory_retirement": {},
        }
    )
    _write_json(receipt_path, v1)
    with pytest.raises(adapter.ContinuationError, match="envelope drifted"):
        adapter._authenticate_sw_always_closure(worker, job=job)


def test_gate_binding_requires_exact_checkpoint_and_both_resume_sidecars(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    worker = adapter.k30._load_worker()
    execution_id = adapter.CONDITIONAL_EXECUTION_IDS[0]
    job = adapter._job_by_id(worker)[execution_id]
    run_root = tmp_path / "runs" / execution_id
    checkpoint = run_root / "checkpoints/current.json"
    ledger = run_root / "checkpoints/current.estimator_call_ledger_checkpoint.a.json"
    verified = run_root / "checkpoints/current.verified_singleton_resume.b.json"
    _write_json(checkpoint, {"round": 30})
    _write_json(ledger, {"ledger": 30})
    _write_json(verified, {"verified": True})
    gate = worker.digested(
        {
            "schema": adapter.k30.PLATEAU_GATE_SCHEMA,
            "status": "passed",
            "execution_id": execution_id,
            "regime_id": job["regime_id"],
            "nph": int(job["nph"]),
            "comparator_policy": job["comparator_policy"],
            "policy": "paper_i_effective_plateau_v1",
            "available_horizon_controller_rounds": 30,
            "extension_decision": (
                "eligible_for_authenticated_resume_to_k50"
            ),
            "source_authorized_horizon": int(job["target_horizon"]),
            "continuation_target_horizon": 50,
            "continuation_materialization_requirement": (
                "authenticated_resume_adapter_only"
            ),
            "resume_checkpoint": {
                "path": "checkpoints/current.json",
                "sha256": worker.sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "resume_checkpoint_siblings": [
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "sha256": worker.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in (ledger, verified)
            ],
            "resume_execution_performed": False,
            "round50_protocol_derived": False,
        }
    )

    observed = adapter._validate_resume_gate_files(
        worker,
        job=job,
        run_root=run_root,
        gate=gate,
    )
    assert observed["resume_checkpoint"]["sha256"] == worker.sha256_file(
        checkpoint
    )

    checkpoint.write_text('{"round":29}\n', encoding="utf-8")
    with pytest.raises(adapter.ContinuationError, match="checkpoint binding"):
        adapter._validate_resume_gate_files(
            worker,
            job=job,
            run_root=run_root,
            gate=gate,
        )


def test_strong_protocol_derivation_changes_only_the_horizon_contract() -> None:
    adapter = _adapter()
    worker = adapter.k30._load_worker()
    execution_id = next(
        row
        for row in adapter.CONDITIONAL_EXECUTION_IDS
        if adapter.k30.NPH_BY_EXECUTION_ID[row] == 7
    )
    jobs = adapter._job_rows_by_id(worker)
    job_path = adapter.PACKAGE_DIR / jobs[execution_id]["job_path"]
    job, _manifest, source, _problem, temporary = worker._prepare(job_path)
    try:
        derived = adapter._derive_strong_k50_protocol(
            worker,
            job=job,
            source_protocol=source,
            continuation_bundle_id="test-continuation-bundle",
            continuation_bundle_manifest_sha256="f" * 64,
        )
    finally:
        temporary.cleanup()

    assert source.horizon == 30
    assert derived.horizon == 50
    assert derived.request.execution.stop.maximum_controller_rounds == 50
    assert derived.stopping_rule["maximum_controller_rounds"] == 50
    assert derived.route_contract == source.route_contract
    assert derived.source_locks == source.source_locks
    assert derived.request.method == source.request.method
    assert adapter._non_horizon_protocol_projection(derived) == (
        adapter._non_horizon_protocol_projection(source)
    )


def test_supervisor_clearance_is_exact_short_lived_and_no_overlap() -> None:
    supervisor = _supervisor()
    now = datetime(2026, 8, 13, 5, 0, tzinfo=timezone.utc)
    execution_ids = ("eligible-a", "eligible-b")
    clearance = {
        "schema": supervisor.REMOTE_CLEARANCE_SCHEMA,
        "status": "passed_authenticated_no_remote_overlap_clearance",
        "execution_ids": list(execution_ids),
        "adapter_sha256": "a" * 64,
        "activation_manifest_sha256": "b" * 64,
        "k30_runtime_manifest_sha256": "c" * 64,
        "observed_at_utc": (now - timedelta(minutes=1)).isoformat(),
        "valid_until_utc": (now + timedelta(minutes=9)).isoformat(),
        "authentication_kind": "interactive_ssh_duo_condor_q_snapshot_v1",
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
        execution_ids=execution_ids,
        adapter_sha256="a" * 64,
        activation_manifest_sha256="b" * 64,
        k30_runtime_manifest_sha256="c" * 64,
        now=now,
    )
    assert observed["no_remote_overlap"] is True

    with pytest.raises(supervisor.SupervisorError, match="clearance drifted"):
        supervisor._validate_remote_overlap_clearance(
            {**clearance, "overlapping_execution_ids": ["eligible-a"]},
            execution_ids=execution_ids,
            adapter_sha256="a" * 64,
            activation_manifest_sha256="b" * 64,
            k30_runtime_manifest_sha256="c" * 64,
            now=now,
        )


def test_supervisor_waits_for_all_nine_decisions_before_any_run() -> None:
    supervisor = _supervisor()
    assert supervisor._sha256_file(supervisor.ADAPTER_PATH) == (
        supervisor.EXPECTED_ADAPTER_SHA256
    )
    pending = {
        "status": "waiting_for_all_k30_decisions",
        "all_decisions_closed": False,
        "pending_execution_ids": ["one"],
        "eligible_execution_ids": ["two"],
        "stop_at_k30_execution_ids": [],
    }
    with pytest.raises(
        supervisor.SupervisorError,
        match="all nine k30 decisions",
    ):
        supervisor._require_all_decisions(pending)


def test_macro_terminal_receipt_requires_every_eligible_k50_cell_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor()
    adapter = supervisor._load_adapter()
    decisions = [
        {"execution_id": execution_id}
        for execution_id in adapter.CONDITIONAL_EXECUTION_IDS
    ]
    eligible = [adapter.CONDITIONAL_EXECUTION_IDS[0]]
    stopped = list(adapter.CONDITIONAL_EXECUTION_IDS[1:])
    snapshot = adapter._digested(
        {
            "schema": adapter.DECISION_STATUS_SCHEMA,
            "status": "passed_all_k30_decisions_closed",
            "all_decisions_closed": True,
            "conditional_execution_ids": list(
                adapter.CONDITIONAL_EXECUTION_IDS
            ),
            "terminal_chtc_k50_execution_ids": list(
                adapter.TERMINAL_CHTC_EXECUTION_IDS
            ),
            "closed_decision_count": 9,
            "pending_execution_ids": [],
            "eligible_execution_ids": eligible,
            "stop_at_k30_execution_ids": stopped,
            "decisions": decisions,
            "scientific_execution_performed": False,
        }
    )
    terminal_status = adapter._digested(
        {
            "schema": adapter.TERMINAL_STATUS_SCHEMA,
            "status": "passed_all_three_authenticated_chtc_k50_terminals",
            "all_terminal_cells_authenticated": True,
            "terminal_chtc_k50_execution_ids": list(
                adapter.TERMINAL_CHTC_EXECUTION_IDS
            ),
            "authenticated_terminal_count": 3,
            "authenticated_terminal_receipts": [
                {"execution_id": execution_id}
                for execution_id in adapter.TERMINAL_CHTC_EXECUTION_IDS
            ],
            "pending_execution_ids": [],
            "validation_errors": {},
            "scientific_execution_performed": False,
        }
    )
    activation = {
        "sha256": "a" * 64,
        "terminal_chtc_k50_execution_ids": list(
            adapter.TERMINAL_CHTC_EXECUTION_IDS
        ),
    }
    runtime = {
        "sha256": "b" * 64,
        "decision_status_sha256": snapshot["sha256"],
        "k30_runtime_manifest_sha256": "c" * 64,
        "conditional_execution_ids": list(adapter.CONDITIONAL_EXECUTION_IDS),
        "eligible_execution_ids": eligible,
        "stop_at_k30_execution_ids": stopped,
        "terminal_chtc_k50_execution_ids": list(
            adapter.TERMINAL_CHTC_EXECUTION_IDS
        ),
    }
    monkeypatch.setattr(
        adapter,
        "closed_continuation_cell",
        lambda *, runtime_dir, execution_id: True,
    )
    path = tmp_path / "terminal.json"

    receipt = supervisor._emit_macro_terminal_receipt(
        adapter,
        runtime=runtime,
        activation=activation,
        snapshot=snapshot,
        terminal_status=terminal_status,
        path=path,
    )

    assert receipt["all_k30_cells_closed"] is True
    assert receipt["all_extension_required_cells_closed_at_k50"] is True
    assert receipt["remaining_macro_execution_ids"] == []
    assert receipt["scientific_execution_performed_by_receipt"] is False
    page12 = _load(
        PAGE12_RUNNER_PATH,
        "paper_i_page12_local_insertion_comparator_macro_gate_test",
    )
    monkeypatch.setattr(
        page12,
        "_trusted_macro_terminal_replay",
        lambda _path: receipt,
    )
    gate_ok, gate_value, gate_blocker = page12._external_gate(
        adapter.k30._load_worker(),
        path,
        kind="macro_terminal",
        now=datetime.now(timezone.utc),
    )
    assert gate_ok is True
    assert gate_value == receipt
    assert gate_blocker is None

    monkeypatch.setattr(
        adapter,
        "closed_continuation_cell",
        lambda *, runtime_dir, execution_id: False,
    )
    with pytest.raises(supervisor.SupervisorError, match="eligible k50"):
        supervisor._emit_macro_terminal_receipt(
            adapter,
            runtime=runtime,
            activation=activation,
            snapshot=snapshot,
            terminal_status=terminal_status,
            path=tmp_path / "must-not-exist.json",
        )
