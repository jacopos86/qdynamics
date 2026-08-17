from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import shutil
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_page12_matched_singleton12_r50_20260815.py"
)
GATE_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "handoff_local_page12_strong5_to_matched_singleton12_20260815.py"
)


def _load_runner():
    name = "paper_i_page12_matched12_runner_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_gate():
    name = "paper_i_page12_matched12_gate_runtime_test"
    spec = importlib.util.spec_from_file_location(name, GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _valid_native_runtime() -> dict:
    return {
        "observed_at_utc": "2026-08-15T00:00:00Z",
        "python": {
            "executable": "/python3",
            "executable_resolved": "/python3",
            "executable_sha256": "a" * 64,
            "version": "3.12.10",
            "implementation": "CPython",
        },
        "packages": {"numpy": "2", "scipy": "1", "qiskit": "2"},
        "loaded_threadpools": [
            {
                "user_api": "blas",
                "internal_api": "accelerate",
                "prefix": "accelerate",
                "filepath": "/System/Accelerate",
                "version": "15.0",
                "num_threads": 1,
                "thread_count_source": "VECLIB_MAXIMUM_THREADS",
            }
        ],
        "loaded_blas_lapack_libraries": [
            "/System/libBLAS",
            "/System/libLAPACK",
        ],
        "numpy_configuration": {"Build Dependencies": {"blas": {"found": True}}},
        "scipy_configuration": {"Build Dependencies": {"lapack": {"found": True}}},
        "cpu": {
            "logical_count": 8,
            "physical_count": 8,
            "brand_string": {"available": False, "value": None},
            "mac_hardware_identity": {
                "available": True,
                "chip_type": "Apple M1 Pro",
                "machine_model": "MacBookPro18,3",
            },
            "numpy_dispatch_features": {"NEON": True},
            "affinity": {"available": False, "cpu_indices": None},
        },
        "platform": {"machine": "arm64", "system": "Darwin"},
        "libc_identity": {
            "platform_libc_ver": {
                "available": False,
                "name": None,
                "version": None,
            },
            "loaded_image_evidence_available": True,
            "loaded_images": [
                {
                    "path": "/usr/lib/libSystem.B.dylib",
                    "version": "26.4",
                    "version_source": (
                        "platform_mac_ver_for_darwin_libsystem_v1"
                    ),
                }
            ],
            "darwin_libsystem_version": "26.4",
        },
        "resource_contract": {
            "kind": "native_local_cpu_only_serial_v1",
            "job_requested_cpu_count": 4,
            "scheduler_allocation_available": False,
            "scheduler_allocated_cpu_count": None,
            "native_local_host_logical_cpu_count": 8,
            "process_affinity_available": False,
            "process_affinity_cpu_count": None,
            "numerical_kernel_thread_count": 1,
            "maximum_campaign_concurrency": 1,
            "gpu_requested_count": 0,
            "gpu_execution_authorized": False,
            "gpu_execution_active": False,
        },
        "capture_point": (
            "inside_cell_after_numpy_scipy_qiskit_blas_load_before_"
            "scientific_execution_v1"
        ),
        "scientific_execution_performed": False,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }


def test_runner_orders_strongest_matched_pairs_first() -> None:
    runner = _load_runner()
    manifest = runner._package_manifest()
    cells = runner._cell_rows(manifest)
    assert [(cell["regime"], cell["n_ph"]) for cell in cells[::2]] == [
        ("strong_strong_u8", 7),
        ("intermediate_strong", 7),
        ("weak_strong", 7),
        ("strong_weak_u8", 3),
        ("intermediate_weak", 3),
        ("weak_weak", 3),
    ]
    assert [cell["method"] for cell in cells[::2]] == [
        "ra_singleton_plateau"
    ] * 6
    assert [cell["method"] for cell in cells[1::2]] == ["append_singleton"] * 6
    archive = runner._archive_module()
    limits = runner._archive_limits(archive)
    assert limits == archive.campaign_default_archive_limits()
    assert runner.MINIMUM_FREE_DISK_BYTES == (
        archive.PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_BYTES
    )


def test_native_runtime_rejects_equal_but_misthreaded_receipts() -> None:
    runner = _load_runner()
    native = _valid_native_runtime()
    assert runner._native_runtime_semantics_valid(native)
    native["loaded_threadpools"][0]["num_threads"] = 2
    assert not runner._native_runtime_semantics_valid(native)


def test_native_runtime_rejects_python_cpu_resource_and_timestamp_tamper() -> None:
    runner = _load_runner()
    native = _valid_native_runtime()
    for mutate in (
        lambda row: row["python"].update(executable_sha256="not-a-digest"),
        lambda row: row["cpu"]["mac_hardware_identity"].update(
            chip_type="arm64"
        ),
        lambda row: row["cpu"].update(numpy_dispatch_features={}),
        lambda row: row["resource_contract"].update(gpu_execution_active=True),
        lambda row: row.update(paper_adoption_authorized=True),
        lambda row: row.update(observed_at_utc="not-a-timestamp"),
    ):
        candidate = copy.deepcopy(native)
        mutate(candidate)
        assert not runner._native_runtime_semantics_valid(candidate)


def test_native_runtime_projection_binds_and_validates_libc_identity() -> None:
    runner = _load_runner()
    native = {
        "schema": "native",
        "sha256": "a" * 64,
        "observed_at_utc": "2026-08-15T00:00:00Z",
        "execution_id": "ra",
        "method": "ra_singleton_plateau",
        "libc_identity": {
            "platform_libc_ver": {
                "available": False,
                "name": None,
                "version": None,
            },
            "loaded_image_evidence_available": True,
            "loaded_images": [
                {
                    "path": "/usr/lib/libSystem.B.dylib",
                    "version": "26.4",
                    "version_source": (
                        "platform_mac_ver_for_darwin_libsystem_v1"
                    ),
                }
            ],
            "darwin_libsystem_version": "26.4",
        },
    }
    matched = copy.deepcopy(native)
    matched.update(
        {
            "sha256": "b" * 64,
            "observed_at_utc": "2026-08-15T00:01:00Z",
            "execution_id": "append",
            "method": "append_singleton",
        }
    )
    assert runner._native_runtime_projection(native) == (
        runner._native_runtime_projection(matched)
    )
    matched["libc_identity"]["loaded_images"][0]["version"] = "26.5"
    assert runner._native_runtime_projection(native) != (
        runner._native_runtime_projection(matched)
    )


def test_pair_runtime_receipt_rejects_redigested_native_digest_tamper() -> None:
    runner = _load_runner()
    ra_native = _valid_native_runtime()
    ra_native.update(
        {
            "schema": (
                "paper_i_page12_matched_singleton12_"
                "native_local_runtime_receipt_v1"
            ),
            "execution_id": "ra-cell",
            "method": "ra_singleton_plateau",
        }
    )
    ra_native = runner._digested(ra_native)
    append_native = copy.deepcopy(ra_native)
    append_native.update(
        {
            "execution_id": "append-cell",
            "method": "append_singleton",
        }
    )
    append_native = runner._digested(
        {key: value for key, value in append_native.items() if key != "sha256"}
    )
    projection = runner._native_runtime_projection(ra_native)
    pair = runner._digested(
        {
            "schema": runner.PAIR_PARITY_SCHEMA,
            "status": "passed_exact_native_runtime_pair_parity",
            "created_at_utc": "2026-08-15T00:02:00Z",
            "regime": "weak_weak",
            "n_ph": 3,
            "ra_execution_id": "ra-cell",
            "append_execution_id": "append-cell",
            "ra_native_runtime_receipt_sha256": ra_native["sha256"],
            "append_native_runtime_receipt_sha256": append_native["sha256"],
            "science_relevant_projection_sha256": (
                runner._canonical_sha256(projection)
            ),
            "allowed_differences": [
                "execution_id",
                "method",
                "observed_at_utc",
                "sha256",
            ],
            "exact_projection_match": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    ra_cell = {
        "execution_id": "ra-cell",
        "method": "ra_singleton_plateau",
        "regime": "weak_weak",
        "n_ph": 3,
    }
    append_cell = {
        "execution_id": "append-cell",
        "method": "append_singleton",
        "regime": "weak_weak",
        "n_ph": 3,
    }
    runner._validate_pair_parity_receipt(
        pair=pair,
        append_cell=append_cell,
        ra_cell=ra_cell,
        append_native=append_native,
        ra_native=ra_native,
    )
    tampered = runner._digested(
        {
            **{key: value for key, value in pair.items() if key != "sha256"},
            "ra_native_runtime_receipt_sha256": "0" * 64,
        }
    )
    with pytest.raises(runner.MatchedSingleton12Error, match="pair runtime"):
        runner._validate_pair_parity_receipt(
            pair=tampered,
            append_cell=append_cell,
            ra_cell=ra_cell,
            append_native=append_native,
            ra_native=ra_native,
        )


def test_guarded_child_inherits_exact_handoff_lock_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    (runtime / "guards").mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_RUNTIME_DIR", runtime)
    captured = {}

    class FakeProcess:
        pid = 123456
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    lock_path = tmp_path / "handoff.lock"
    with lock_path.open("w+") as lock:
        os.set_inheritable(lock.fileno(), True)
        monkeypatch.setenv(runner.HANDOFF_LOCK_FD_ENV, str(lock.fileno()))
        receipt = runner._run_guarded_child(
            cell={"execution_id": "cell", "max_runtime_seconds": 10},
            activation={"sha256": "a" * 64},
            handoff={"sha256": "b" * 64},
        )
        assert captured["pass_fds"] == (lock.fileno(),)
    assert receipt["status"] == "passed"


def test_activation_digest_chain_rejects_redigested_authority_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    manifest = {"sha256": "m" * 64}
    cell = {
        "execution_id": "cell",
        "method": "ra_singleton_plateau",
        "regime": "weak_weak",
        "n_ph": 3,
        "job_path": "/job",
        "job_spec_sha256": "j" * 64,
        "protocol_sha256": "q" * 64,
        "route_contract_sha256": "z" * 64,
        "max_runtime_seconds": 100,
    }
    plan = {
        "schema": runner.PLAN_SCHEMA,
        "status": "passed_strongest_pairs_first_serial_plan",
        "package_manifest_sha256": manifest["sha256"],
        "cells": [cell],
        "execution_ids": ["cell"],
        "sha256": "p" * 64,
    }
    planning = {
        "schema": runner.PLANNING_SCHEMA,
        "status": "passed_inert_planning",
        "execution_plan": "plan-binding",
        "package_manifest": "package-binding",
        "runner": "runner-binding",
        "worker": "worker-binding",
        "archive_module": "archive-binding",
        "maximum_concurrency": 1,
        "execution_authorized": False,
        "submission_authorized": False,
        "paper_evidence_adoption_authorized": False,
        "sha256": "l" * 64,
    }
    runtime = {"schema": runner.RUNTIME_FINGERPRINT_SCHEMA, "sha256": "r" * 64}
    parity = {
        "schema": runner.PARITY_SCHEMA,
        "status": runner.PARITY_STATUS,
        "package_manifest_sha256": manifest["sha256"],
        "sealed_ra_protocol_sha256_by_regime": {"weak_weak": "q" * 64},
        "checkpoint_overlay_parity_canary": "checkpoint-binding",
        "sha256": "c" * 64,
    }
    authorization = {
        "schema": runner.AUTHORIZATION_SCHEMA,
        "status": "authorized_local_matched_singleton12_execution",
        "authorized_by": "user_confirmed_local_paper_i_singleton_rerun",
        "planning_manifest_sha256": planning["sha256"],
        "execution_plan_sha256": plan["sha256"],
        "package_manifest_sha256": manifest["sha256"],
        "runtime_fingerprint_sha256": "WRONG",
        "scientific_parity_canary_sha256": parity["sha256"],
        "execution_ids": ["cell"],
        "execution_authorized": True,
        "archive_rotation_authorized": True,
        "submission_authorized": False,
        "paper_evidence_adoption_authorized": False,
        "sha256": "u" * 64,
    }
    activation = {
        "schema": runner.ACTIVATION_SCHEMA,
        "status": "authorized_local_execution",
        "package_manifest_sha256": manifest["sha256"],
        "planning_manifest_sha256": planning["sha256"],
        "execution_plan_sha256": plan["sha256"],
        "execution_authorization_sha256": authorization["sha256"],
        "scientific_parity_canary_sha256": parity["sha256"],
        "runtime_fingerprint_sha256": runtime["sha256"],
        "runner": "runner-binding",
        "worker": "worker-binding",
        "archive_module": "archive-binding",
        "execution_ids": ["cell"],
        "execution_authorized": True,
        "archive_rotation_authorized": True,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
        "sha256": "a" * 64,
    }
    documents = {
        "planning_manifest.json": planning,
        "execution_plan.json": plan,
        "execution_authorization.json": authorization,
        "activation_manifest.json": activation,
        "scientific_parity_canary.json": parity,
        "runtime_fingerprint.json": runtime,
    }
    monkeypatch.setattr(
        runner,
        "_load_digested",
        lambda path, **_kwargs: documents[Path(path).name],
    )
    monkeypatch.setattr(runner, "_package_manifest", lambda: manifest)
    monkeypatch.setattr(runner, "_cell_rows", lambda _manifest: [cell])
    monkeypatch.setattr(runner, "_live_runtime_fingerprint", lambda: runtime)

    def fake_binding(path, **_kwargs):
        if Path(path) == runner.RUNNER_PATH:
            return "runner-binding"
        if Path(path) == runner.WORKER_PATH:
            return "worker-binding"
        return "archive-binding"

    monkeypatch.setattr(runner, "_binding", fake_binding)

    def fake_validate_binding(_raw, *, label, **_kwargs):
        if label == "planning execution plan":
            return tmp_path / "plan", plan
        if label == "planning package manifest":
            return tmp_path / "package", manifest
        return (
            runner.REPAIR_ROOT
            / (
                "paper_i_page12_strong_holstein_sector5_local_repair_"
                "20260814_v1_activation/scientific_parity_canary.json"
            ),
            {"sha256": "ad870ca15fd75b31400986c71245a56283532a5b5714b1c456185ce87ad0ceaa"},
        )

    monkeypatch.setattr(runner, "_validate_binding", fake_validate_binding)
    with pytest.raises(runner.MatchedSingleton12Error, match="Activation authority"):
        runner._validate_activation(tmp_path)


def test_planning_publish_recovers_from_staged_mid_write_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    planning_dir = tmp_path / "planning"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    manifest = {"sha256": "m" * 64}
    cells = [
        {
            "execution_id": f"cell-{index}",
            "method": (
                "ra_singleton_plateau" if index % 2 == 0 else "append_singleton"
            ),
            "regime": f"regime-{index // 2}",
            "n_ph": 7 if index < 6 else 3,
            "job_path": f"/job-{index}.json",
            "job_spec_sha256": f"{index:064x}",
            "protocol_sha256": f"{index + 20:064x}",
            "route_contract_sha256": f"{index + 40:064x}",
            "max_runtime_seconds": 259_200,
        }
        for index in range(12)
    ]

    class FakeLimits:
        archive_start_free_floor_bytes = 10

        def as_dict(self):
            return {
                "max_member_payload_bytes": 1,
                "max_total_payload_bytes": 2,
                "max_decompressed_bytes": 3,
                "max_compressed_bytes": 4,
                "min_free_disk_bytes": 5,
            }

    class FakeArchive:
        ArchiveLimits = FakeLimits

        @staticmethod
        def campaign_default_archive_limits():
            return FakeLimits()

        @staticmethod
        def require_campaign_capacity(_path):
            return {"status": "passed"}

    class FakeWorker:
        @staticmethod
        def preflight(job_path):
            return {
                "status": "passed",
                "job_path": str(job_path),
                "scientific_execution_performed": False,
                "execution_source_policy": runner.EXECUTION_SOURCE_POLICY,
                "fresh_start_only": True,
                "checkpoint_usage": runner.CHECKPOINT_USAGE,
                "checkpoint_resume_authorized": False,
                "checkpoint_overlay": {
                    "ambient_resume_overlay": False,
                    "sealed_resume_reader_sha256": (
                        runner.SEALED_RESUME_READER_SHA256
                    ),
                },
            }

    monkeypatch.setattr(runner, "_package_manifest", lambda: manifest)
    monkeypatch.setattr(runner, "_cell_rows", lambda _manifest: cells)
    monkeypatch.setattr(runner, "_archive_module", lambda: FakeArchive)
    monkeypatch.setattr(runner, "_load_module", lambda *_args: FakeWorker)
    monkeypatch.setattr(
        runner,
        "_capacity",
        lambda _path: {
            "available_memory_bytes": runner.MINIMUM_AVAILABLE_MEMORY_BYTES,
            "free_disk_bytes": runner.MINIMUM_FREE_DISK_BYTES,
        },
    )
    original_write = runner._write_json_exclusive
    failed_once = False

    def fail_during_staged_publish(path, payload):
        nonlocal failed_once
        if path.name == "inert_preflight.json" and not failed_once:
            failed_once = True
            raise OSError("injected planning crash")
        original_write(path, payload)

    monkeypatch.setattr(runner, "_write_json_exclusive", fail_during_staged_publish)
    with pytest.raises(OSError, match="injected planning crash"):
        runner.prepare_planning(
            planning_dir=planning_dir, runtime_dir=runtime_dir
        )
    assert not planning_dir.exists()
    assert planning_dir.with_name(f".{planning_dir.name}.in_progress").is_dir()

    monkeypatch.setattr(runner, "_write_json_exclusive", original_write)
    planning = runner.prepare_planning(
        planning_dir=planning_dir, runtime_dir=runtime_dir
    )
    validated, plan = runner._validated_planning(planning_dir)
    assert planning == validated
    assert plan["execution_ids"] == [cell["execution_id"] for cell in cells]
    assert not planning_dir.with_name(f".{planning_dir.name}.in_progress").exists()
    assert list(tmp_path.glob(".planning.quarantine.*"))


def test_terminal_promise_is_replayable_until_terminal_publishes_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    gate = _load_gate()
    runtime = tmp_path / "runtime"
    (runtime / "status").mkdir(parents=True)
    monkeypatch.setattr(runner, "DEFAULT_RUNTIME_DIR", runtime)
    execution_ids = ["ra-cell", "append-cell"]
    terminal = runner._digested(
        {
            "schema": runner.TERMINAL_SCHEMA,
            "status": runner.TERMINAL_STATUS,
            "completed_at_utc": "2026-08-15T12:00:00Z",
            "execution_ids": execution_ids,
            "completed_execution_ids": execution_ids,
        }
    )
    status = runner._terminal_status_receipt(
        terminal=terminal,
        execution_ids=execution_ids,
        updated_at_utc="2026-08-15T12:00:01Z",
    )
    original_terminal_write = runner._write_json_atomic_noreplace

    class SimulatedHardKill(BaseException):
        pass

    def crash_before_terminal(_path, _payload):
        raise SimulatedHardKill()

    monkeypatch.setattr(
        runner, "_write_json_atomic_noreplace", crash_before_terminal
    )
    with pytest.raises(SimulatedHardKill):
        runner._publish_terminal_last(terminal=terminal, status=status)
    assert not (runtime / "terminal_receipt.json").exists()
    assert runner._load_terminal_publication_promise(execution_ids) == status
    target = {
        "runtime_dir": runtime.as_posix(),
        "expected_terminal": {
            "path": (runtime / "terminal_receipt.json").as_posix(),
            "schema": runner.TERMINAL_SCHEMA,
            "status": runner.TERMINAL_STATUS,
        },
    }
    assert gate._validate_target_runtime_state(
        target, claim_exists=True
    ) == {"state": "replayable_after_claim"}

    monkeypatch.setattr(
        runner, "_write_json_atomic_noreplace", original_terminal_write
    )
    runner._publish_terminal_last(terminal=terminal, status=status)
    assert runner._load_digested(
        runtime / "terminal_receipt.json", label="test terminal"
    ) == terminal
    final_status = runner._load_digested(
        runtime / "status/campaign.json", label="test terminal status"
    )
    assert final_status == status
    assert final_status["terminal_receipt_sha256"] == terminal["sha256"]
    state = gate._validate_target_runtime_state(target, claim_exists=True)
    assert state["state"] == "complete"
    assert state["terminal"] == terminal


def test_runtime_root_publish_recovers_from_staged_manifest_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(runner, "DEFAULT_RUNTIME_DIR", runtime)
    activation = {"sha256": "a" * 64}
    handoff = {"sha256": "h" * 64}
    original_write = runner._write_json_exclusive
    failed_once = False

    def fail_runtime_manifest(path, payload):
        nonlocal failed_once
        if path.name == "runtime_manifest.json" and not failed_once:
            failed_once = True
            raise OSError("injected runtime manifest crash")
        original_write(path, payload)

    monkeypatch.setattr(runner, "_write_json_exclusive", fail_runtime_manifest)
    with pytest.raises(OSError, match="injected runtime manifest crash"):
        runner._ensure_runtime(activation, handoff)
    assert not runtime.exists()
    assert runtime.with_name(f".{runtime.name}.in_progress").is_dir()

    monkeypatch.setattr(runner, "_write_json_exclusive", original_write)
    published = runner._ensure_runtime(activation, handoff)
    assert runner._load_digested(
        runtime / "runtime_manifest.json", label="test runtime manifest"
    ) == published
    assert {
        "runs",
        "receipts",
        "runtime_checks",
        "guards",
        "pair_parity",
        "status",
    }.issubset({path.name for path in runtime.iterdir() if path.is_dir()})
    assert not runtime.with_name(f".{runtime.name}.in_progress").exists()
    assert list(tmp_path.glob(".runtime.quarantine.*"))


def test_pre_cell_runtime_check_refreshes_only_while_science_is_unbound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    runtime = tmp_path / "runtime"
    for name in ("runtime_checks", "runs", "receipts", "guards"):
        (runtime / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "DEFAULT_RUNTIME_DIR", runtime)
    recorded = {"sha256": "r" * 64}
    monkeypatch.setattr(runner, "_live_runtime_fingerprint", lambda: recorded)
    monkeypatch.setattr(
        runner,
        "_capacity",
        lambda _path: {
            "available_memory_bytes": runner.MINIMUM_AVAILABLE_MEMORY_BYTES,
            "free_disk_bytes": runner.MINIMUM_FREE_DISK_BYTES,
        },
    )

    class FakeArchive:
        @staticmethod
        def require_regime_launch_capacity(_path, *, regime_id, nph):
            return {"status": "passed", "regime_id": regime_id, "nph": nph}

    monkeypatch.setattr(runner, "_archive_module", lambda: FakeArchive)
    cell = {
        "execution_id": "cell",
        "regime": "weak_weak",
        "n_ph": 3,
    }
    first = runner._runtime_check(recorded, cell)
    second = runner._runtime_check(recorded, cell)
    assert second["replaces_runtime_check_sha256"] == first["sha256"]
    path = runtime / "runtime_checks/cell.json"
    assert runner._load_digested(path, label="refreshed runtime check") == second

    (runtime / "runs/cell").mkdir()
    with pytest.raises(runner.MatchedSingleton12Error, match="science evidence"):
        runner._runtime_check(recorded, cell)
    assert runner._load_digested(path, label="bound runtime check") == second


def _runner_archive_fixture(tmp_path: Path, runner):
    module = runner._archive_module()
    execution = "runner_archive_restart_cell"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    paths = module.CellArchivePaths(runtime_root=runtime, execution_id=execution)
    (paths.source_root / "runtime").mkdir(parents=True)
    (paths.source_root / "runtime/native_runtime.json").write_text(
        '{"sha256":"runtime","status":"passed"}\n', encoding="utf-8"
    )
    (paths.source_root / "result.json").write_text(
        '{"status":"passed"}\n', encoding="utf-8"
    )
    external_root = tmp_path / "external"
    external_root.mkdir()
    external = external_root / "worker.json"
    external.write_text('{"status":"passed"}\n', encoding="utf-8")
    external_members = {"evidence/worker.json": external}
    authority = {
        "activation_manifest_sha256": "a" * 64,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    cell = {
        "execution_id": execution,
        "method": "ra_singleton_plateau",
        "regime": "weak_weak",
    }
    rotation = {
        "execution_authorized": True,
        "archive_rotation_authorized": True,
        "authorization_sha256": "b" * 64,
    }
    limits = module.ArchiveLimits(
        max_member_payload_bytes=2 * 1024 * 1024,
        max_total_payload_bytes=4 * 1024 * 1024,
        max_decompressed_bytes=8 * 1024 * 1024,
        max_compressed_bytes=8 * 1024 * 1024,
        min_free_disk_bytes=0,
    )
    return (
        module,
        paths,
        f"runs/{execution}",
        external_members,
        authority,
        cell,
        rotation,
        limits,
    )


def _drive_runner_archive(runner, fixture):
    module, paths, prefix, external, authority, cell, rotation, limits = fixture
    return runner._drive_archive_rotation_state(
        module=module,
        paths=paths,
        state=module.inspect_rotation_state(paths),
        source_member_prefix=prefix,
        external_members=external,
        authority_metadata=authority,
        cell_metadata=cell,
        rotation_authority=rotation,
        limits=limits,
    )


def test_runner_restart_validates_archived_cell_with_source_absent(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    fixture = _runner_archive_fixture(tmp_path, runner)
    first = _drive_runner_archive(runner, fixture)
    module, paths, *_rest = fixture
    assert module.inspect_rotation_state(paths)["state"] == "archived_closed"
    assert not paths.source_root.exists()
    second = _drive_runner_archive(runner, fixture)
    assert second == first


def test_runner_archived_restart_rejects_tampered_archive(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    fixture = _runner_archive_fixture(tmp_path, runner)
    _drive_runner_archive(runner, fixture)
    module, paths, *_rest = fixture
    with paths.archive_path.open("r+b") as stream:
        stream.seek(max(1, paths.archive_path.stat().st_size // 2))
        original = stream.read(1)
        stream.seek(-1, os.SEEK_CUR)
        stream.write(bytes([original[0] ^ 0x01]))
    with pytest.raises(module.Singleton12ArchiveError):
        _drive_runner_archive(runner, fixture)


@pytest.mark.parametrize(
    "state_name",
    ["retiring_pending_removal", "cleanup_receipt_pending"],
)
def test_runner_resumes_rotation_only_crash_states(
    tmp_path: Path, state_name: str
) -> None:
    runner = _load_runner()
    fixture = _runner_archive_fixture(tmp_path, runner)
    module, paths, prefix, external, authority, cell, rotation, limits = fixture
    module.build_cell_archive(
        paths=paths,
        source_member_prefix=prefix,
        external_members=external,
        authority_metadata=authority,
        cell_metadata=cell,
        limits=limits,
    )
    module.publish_archive_closure(
        paths=paths,
        source_member_prefix=prefix,
        authority_metadata=authority,
        cell_metadata=cell,
        limits=limits,
    )
    module.publish_rotation_intent(
        paths=paths,
        source_member_prefix=prefix,
        authority_metadata=authority,
        cell_metadata=cell,
        rotation_authority=rotation,
        limits=limits,
    )
    paths.retiring_root.parent.mkdir(exist_ok=True)
    os.rename(paths.source_root, paths.retiring_root)
    if state_name == "cleanup_receipt_pending":
        shutil.rmtree(paths.retiring_root)
    assert module.inspect_rotation_state(paths)["state"] == state_name
    closure = _drive_runner_archive(runner, fixture)
    assert closure["status"] == "passed_archive_backed_terminal_closure"
    assert module.inspect_rotation_state(paths)["state"] == "archived_closed"


@pytest.mark.parametrize(
    "stage",
    [
        "direct_unarchived",
        "archive_published_pending_manifest",
        "manifest_published_pending_closure",
        "closure_published_pending_intent",
        "intent_published_pending_rename",
    ],
)
def test_runner_resumes_each_source_preserving_archive_stage(
    tmp_path: Path, stage: str
) -> None:
    runner = _load_runner()
    fixture = _runner_archive_fixture(tmp_path, runner)
    module, paths, prefix, external, authority, cell, rotation, limits = fixture
    if stage != "direct_unarchived":
        module.build_cell_archive(
            paths=paths,
            source_member_prefix=prefix,
            external_members=external,
            authority_metadata=authority,
            cell_metadata=cell,
            limits=limits,
        )
    if stage == "archive_published_pending_manifest":
        paths.archive_manifest_path.unlink()
    elif stage in {
        "closure_published_pending_intent",
        "intent_published_pending_rename",
    }:
        module.publish_archive_closure(
            paths=paths,
            source_member_prefix=prefix,
            authority_metadata=authority,
            cell_metadata=cell,
            limits=limits,
        )
        if stage == "intent_published_pending_rename":
            module.publish_rotation_intent(
                paths=paths,
                source_member_prefix=prefix,
                authority_metadata=authority,
                cell_metadata=cell,
                rotation_authority=rotation,
                limits=limits,
            )
    if stage == "direct_unarchived":
        temporary = paths.archive_path.with_name(
            f".{paths.archive_path.name}.tmp.{'0' * 32}"
        )
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"stale-build")
    assert module.inspect_rotation_state(paths)["state"] == stage
    closure = _drive_runner_archive(runner, fixture)
    assert closure["status"] == "passed_archive_backed_terminal_closure"
    assert module.inspect_rotation_state(paths)["state"] == "archived_closed"
