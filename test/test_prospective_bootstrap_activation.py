"""Focused local-only contracts for prospective exact/scout activation."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
from typing import Any, Mapping

import pytest

import pipelines.time_dynamics.prospective_bootstrap_activation as activation_module
import pipelines.time_dynamics.prospective_exact_reference_worker as exact_worker
import pipelines.time_dynamics.runners.build_prospective_exact_reference as exact_runner
from pipelines.time_dynamics.prospective_avqds_scout_worker import (
    build_work_order as build_avqds_work_order,
    load_frozen_submitted_work_order as load_frozen_avqds_work_order,
    write_work_order as write_avqds_work_order,
)
from pipelines.time_dynamics.prospective_avqds_scout_chtc_package import (
    AVQDSCHTCResources,
    AVQDSPrefixPackageError,
    avqds_scout_submission_unit_id,
    materialize_avqds_prefix_package,
    validate_avqds_prefix_package,
    validate_frozen_submitted_avqds_prefix_package,
)
from pipelines.time_dynamics.prospective_avqds_runtime_source import (
    SCOUT_PROFILE,
    materialize_avqds_runtime_source_archive,
)
from pipelines.time_dynamics.combined_scheduler_snapshot import (
    CAPTURE_SPEC_V1,
    HH_ACQUISITION_LANE_ID,
    HUBBARD_ACQUISITION_LANE_ID,
    build_combined_snapshot,
    write_immutable_snapshot,
)
from pipelines.time_dynamics.prospective_bootstrap_activation import (
    BootstrapActivationError,
    authorize_from_snapshot,
    load_activation_manifest,
    load_avqds_scout_terminal_failure_evidence,
    load_exact_reference_terminal_failure_evidence,
    load_submission_intent,
    load_submission_receipt,
    prepare_bootstrap_activation,
    prepare_avqds_scout_terminal_failure_successor,
    prepare_exact_reference_terminal_failure_successor,
    record_dry_run_result,
    record_exact_reference_terminal_failure_evidence,
    record_provisional_submit_result,
    record_upload_verification,
    recover_submission_receipt,
    snapshot_intent_binding,
)
from pipelines.time_dynamics.prospective_exact_reference_chtc_package import (
    PreparedExactReferenceActionPackage,
    materialize_exact_reference_action_package,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import LANE_HUBBARD
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    write_runtime_image_identity,
)
from pipelines.time_dynamics.runners.coordinate_prospective_bootstrap_activation import (
    main as activation_cli,
)


ATTRIBUTES = (
    "ClusterId,ProcId,JobStatus,ExitCode,HoldReason,JobBatchName,"
    "AcquisitionUnitId,AcquisitionLaneId,PaperIIActionId,"
    "PaperIIExecutionRequestSha256,PaperIISourceEpochId,"
    "EnteredCurrentStatus,CompletionDate,QDate"
)


@dataclass(frozen=True)
class SnapshotFiles:
    path: Path
    payload: dict[str, Any]
    queue_stdout: Path
    history_stdout: Path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=root, text=True
    ).strip()


def _fixture_builder():
    path = Path(__file__).with_name("test_prospective_avqds_scout_worker.py")
    spec = importlib.util.spec_from_file_location(
        "_prospective_activation_fixture_source", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the qualified-seed test fixture")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._receipts_and_request


@pytest.fixture(scope="module")
def prepared_repository(tmp_path_factory: pytest.TempPathFactory) -> Mapping[str, Any]:
    root = tmp_path_factory.mktemp("bootstrap-activation") / "repository"
    root.mkdir()
    source, seed, scout_request = _fixture_builder()(root)

    overlays = {
        "pipelines/time_dynamics/prospective_exact_reference_worker.py": Path(
            exact_worker.__file__
        ),
        "pipelines/time_dynamics/runners/build_prospective_exact_reference.py": Path(
            exact_runner.__file__
        ),
    }
    for relative, source_file in overlays.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
    _git(root, "add", *overlays)
    _git(root, "commit", "-qm", "committed exact-reference executor overlay")

    source_tar = subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source.source_epoch.source_revision,
            "--",
            "pipelines",
            "src",
            "requirements.txt",
        ),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as handle:
        handle.write(source_tar)
    exact_archive = root / "runtime/exact_source.tar.gz"
    exact_archive.parent.mkdir(parents=True, exist_ok=True)
    exact_archive.write_bytes(compressed.getvalue())
    scout_order = build_avqds_work_order(
        scout_request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=root,
    )
    avqds_archive = root / "runtime/avqds_exact_blind_source.tar.gz"
    materialize_avqds_runtime_source_archive(
        avqds_archive,
        repository_root=root,
        profile=SCOUT_PROFILE,
        base_revision=str(source.source_epoch.source_revision),
        base_full_archive_sha256=str(source.source_epoch.source_archive_sha256),
        overlay_revision=scout_order.executor_overlay.overlay_revision,
        overlay_files=scout_order.executor_overlay.files,
    )
    image = root / "runtime/paper_ii_runtime.sif"
    image.write_bytes(b"test Paper-II runtime image\n")
    image_identity = root / "runtime/paper_ii_runtime.identity.json"
    write_runtime_image_identity(
        image_identity, image_path=image, repository_root=root
    )

    manifest = prepare_bootstrap_activation(
        repository_root=root,
        output_dir="prepared/bootstrap_activation",
        seed_receipt_paths=[seed.receipt_path],
        source_epoch_receipt_paths=[source.receipt_path],
        exact_source_archive_paths={
            LANE_HUBBARD: exact_archive.relative_to(root).as_posix()
        },
        avqds_source_archive_paths={
            LANE_HUBBARD: avqds_archive.relative_to(root).as_posix()
        },
        runtime_image_identity_path=image_identity.relative_to(root).as_posix(),
        free_bytes=30 * 1024**3,
        recorded_at="2026-08-29T02:00:00-05:00",
    )
    return {
        "root": root,
        "manifest": manifest,
        "seed_receipt": seed.receipt_path,
        "source_receipt": source.receipt_path,
        "exact_source_archive": exact_archive.relative_to(root).as_posix(),
        "avqds_source_archive": avqds_archive.relative_to(root).as_posix(),
        "image_identity": image_identity.relative_to(root).as_posix(),
    }


@pytest.fixture
def repository(
    tmp_path: Path, prepared_repository: Mapping[str, Any]
) -> tuple[Path, Mapping[str, Any]]:
    root = tmp_path / "repository"
    shutil.copytree(prepared_repository["root"], root)
    manifest = load_activation_manifest(
        "prepared/bootstrap_activation/activation_manifest.json",
        repository_root=root,
    )
    return root, manifest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _snapshot(
    root: Path,
    *,
    intent_path: str,
    name: str,
    captured_at: str,
    captured_at_unix_ns: int,
    rows: list[dict[str, object]] | None = None,
    history_rows: list[dict[str, object]] | None = None,
    prior: Mapping[str, Any] | None = None,
    empty_successful_streams: bool = False,
) -> SnapshotFiles:
    intent = load_submission_intent(intent_path, repository_root=root)
    batch = intent.package_manifest.job_batch_name
    queue_rows = rows or []
    terminal_rows = history_rows or []
    scheduler_rows = [*queue_rows, *terminal_rows]
    cluster_ids = sorted(
        {int(row["ClusterId"]) for row in scheduler_rows}
    )
    condition_parts = [f'JobBatchName == "{batch}"']
    condition_parts.extend(f"ClusterId == {cluster}" for cluster in cluster_ids)
    condition = " || ".join(f"({part})" for part in condition_parts)
    lane_key = "hh" if intent.lane_id == HH_ACQUISITION_LANE_ID else "hubbard"
    lane_scopes = {
        "hh": {
            "cluster_ids": [],
            "job_batch_names": [],
            "acquisition_lane_ids": [HH_ACQUISITION_LANE_ID],
        },
        "hubbard": {
            "cluster_ids": [],
            "job_batch_names": [],
            "acquisition_lane_ids": [HUBBARD_ACQUISITION_LANE_ID],
        },
        "siam": {
            "cluster_ids": [],
            "job_batch_names": [],
            "acquisition_lane_ids": [],
        },
    }
    lane_scopes[lane_key]["cluster_ids"] = cluster_ids
    lane_scopes[lane_key]["job_batch_names"] = [batch]
    query_argv = {
        "queue": [
            "condor_q",
            "jsstrobel",
            "-constraint",
            condition,
            "-json",
            "-attributes",
            ATTRIBUTES,
        ],
        "history": [
            "condor_history",
            "jsstrobel",
            "-constraint",
            condition,
            "-json",
            "-attributes",
            ATTRIBUTES,
        ],
    }
    queue_path = root / "snapshots" / f"{name}.queue.json"
    history_path = root / "snapshots" / f"{name}.history.json"
    _write_json(queue_path, queue_rows)
    _write_json(history_path, terminal_rows)
    if empty_successful_streams:
        queue_path.write_bytes(b"")
        history_path.write_bytes(b"")
    snapshot = build_combined_snapshot(
        queue_stdout=queue_path,
        history_stdout=history_path,
        capture_spec={
            "schema": CAPTURE_SPEC_V1,
            "captured_at": captured_at,
            "captured_at_unix_ns": captured_at_unix_ns,
            "single_combined_snapshot": True,
            "polling_performed": False,
            "capture_counts": {"queue": 1, "history": 1},
            "reconciliation_intents": {batch: snapshot_intent_binding(intent)},
            "prior_snapshot": prior,
            "scope": {"condition": condition, "lanes": lane_scopes},
            "queries": {
                query: {"argv": argv, "returncode": 0, "stderr": ""}
                for query, argv in query_argv.items()
            },
        },
    )
    output = root / "snapshots" / f"{name}.json"
    assert write_immutable_snapshot(output, snapshot) == "created"
    return SnapshotFiles(output, snapshot, queue_path, history_path)


def _found_row(intent_path: str, root: Path, *, cluster_id: int) -> dict[str, object]:
    intent = load_submission_intent(intent_path, repository_root=root)
    return {
        "ClusterId": cluster_id,
        "ProcId": 0,
        "JobStatus": 1,
        "JobBatchName": intent.package_manifest.job_batch_name,
        "AcquisitionUnitId": intent.unit.unit_id,
        "AcquisitionLaneId": intent.lane_id,
        "PaperIIActionId": intent.request.action_id,
        "PaperIIExecutionRequestSha256": intent.request.digest,
        "PaperIISourceEpochId": intent.source_epoch.source_epoch.source_epoch_id,
        "EnteredCurrentStatus": 1,
        "QDate": 1,
    }


def _exact_intent(manifest: Mapping[str, Any]) -> str:
    return str(manifest["pairs"][0]["exact_reference"]["intent"]["path"])


def _avqds_intent(manifest: Mapping[str, Any]) -> str:
    return str(manifest["pairs"][0]["avqds_scout"]["intent"]["path"])


def _dry_run_artifacts(
    root: Path, intent_path: str, *, name: str = "dry-run"
) -> tuple[Path, Path]:
    intent = load_submission_intent(intent_path, repository_root=root)
    capture = root / "preserved_scheduler_evidence" / name
    capture.mkdir(parents=True, exist_ok=True)
    ad_path = capture / "dryrun.ad"
    ad_path.write_bytes(b"preserved deterministic dry-run ClassAd\n")
    descriptor_path = capture / "submit.sub"
    shutil.copyfile(
        root / intent.unit.payload["submit_descriptor"], descriptor_path
    )
    return ad_path, descriptor_path


def _first_generation_authorization(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    prefix: str,
) -> tuple[str, Any, SnapshotFiles, SnapshotFiles, Mapping[str, Any]]:
    intent_path = _exact_intent(manifest)
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(
        root, intent_path, name=f"{prefix}-dry-run"
    )
    record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )
    first = _snapshot(
        root,
        intent_path=intent_path,
        name=f"{prefix}-first",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
    )
    absent = _snapshot(
        root,
        intent_path=intent_path,
        name=f"{prefix}-absent",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=first.payload,
    )
    authorization = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=absent.path,
        queue_stdout_path=absent.queue_stdout,
        history_stdout_path=absent.history_stdout,
        prior_snapshot_path=first.path,
        prior_queue_stdout_path=first.queue_stdout,
        prior_history_stdout_path=first.history_stdout,
    )
    return intent_path, intent, first, absent, authorization


def _terminal_row(
    intent_path: str,
    root: Path,
    *,
    cluster_id: int,
    job_status: int = 4,
    exit_code: int = 1,
) -> dict[str, object]:
    row = _found_row(intent_path, root, cluster_id=cluster_id)
    row.update(
        {
            "JobStatus": job_status,
            "ExitCode": exit_code,
            "CompletionDate": 300,
        }
    )
    return row


def _diagnostic_archive(
    root: Path,
    *,
    intent_path: str,
    cluster_id: int,
    name: str,
) -> Path:
    intent = load_submission_intent(intent_path, repository_root=root)
    prefix = str(intent.request.inputs["reference_identity_sha256"])[:16]
    archive = root / "failure_diagnostics" / f"{name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = {
        (
            f"paper_ii_hubbard/{prefix}/exact_reference/raw_outputs/"
            "paper_ii_exact_reference/action/worker_stderr.log"
        ): (
            b"Traceback (most recent call last):\n"
            b"  request = ExecutionRequest.from_mapping(request_row)\n"
            b"ProspectiveExecutionError: execution request resolver binding mismatch\n"
        ),
        (
            f"paper_ii_hubbard/{prefix}/exact_reference/logs/"
            f"reference.{cluster_id}.0.log"
        ): (
            f"000 ({cluster_id}.000.000) submitted\n"
            f"005 ({cluster_id}.000.000) Normal termination (return value 1)\n"
        ).encode("utf-8"),
    }
    with tarfile.open(archive, "w:gz") as handle:
        for path, content in sorted(members.items()):
            info = tarfile.TarInfo(path)
            info.size = len(content)
            info.mode = 0o600
            info.mtime = 0
            handle.addfile(info, io.BytesIO(content))
    return archive


def _submitted_terminal_predecessor(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    prefix: str,
    history_rows: list[dict[str, object]] | None = None,
) -> tuple[str, Any, Path, Path, SnapshotFiles, Path]:
    intent_path, intent, _first, _absent, authorization = (
        _first_generation_authorization(root, manifest, prefix=prefix)
    )
    cluster_id = 777
    record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=authorization["authorization_path"],
        returncode=0,
        stdout=(
            "Submitting job(s).\n"
            f"1 job(s) submitted to cluster {cluster_id}.\n"
        ),
        submitted_at="2026-08-29T02:05:00-05:00",
    )
    rows = history_rows
    if rows is None:
        rows = [
            _terminal_row(
                intent_path, root, cluster_id=cluster_id
            )
        ]
    terminal = _snapshot(
        root,
        intent_path=intent_path,
        name=f"{prefix}-terminal",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=300,
        history_rows=rows,
    )
    result_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor/submission_results"
        / f"{intent.unit.unit_id}.json"
    )
    recover_submission_receipt(
        repository_root=root,
        intent_path=intent_path,
        provisional_result_path=result_path,
        later_snapshot_path=terminal.path,
        queue_stdout_path=terminal.queue_stdout,
        history_stdout_path=terminal.history_stdout,
    )
    receipt_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor/submission_receipts"
        / f"{intent.unit.unit_id}.json"
    )
    archive = _diagnostic_archive(
        root,
        intent_path=intent_path,
        cluster_id=cluster_id,
        name=prefix,
    )
    return intent_path, intent, result_path, receipt_path, terminal, archive


def test_prepare_materializes_independent_exact_and_scout_arms(
    repository: tuple[Path, Mapping[str, Any]],
    prepared_repository: Mapping[str, Any],
) -> None:
    root, manifest = repository
    assert manifest["pair_count"] == 2
    assert manifest["action_count"] == 4
    assert manifest["scheduler_contacted"] is False
    assert manifest["production_submission_enabled"] is False
    assert manifest["qiskit_compilation"] is False
    assert manifest["concurrency_contract"] == {
        "exact_and_avqds_launch_independently": True,
        "join_only_after_completion": True,
        "siam_dependency": False,
        "apm_selected": False,
    }
    for pair in manifest["pairs"]:
        assert pair["concurrency"]["join_required_before_launch"] is False
        assert pair["concurrency"]["siam_dependency"] is False
        exact = load_submission_intent(
            pair["exact_reference"]["intent"]["path"], repository_root=root
        )
        scout = load_submission_intent(
            pair["avqds_scout"]["intent"]["path"], repository_root=root
        )
        assert exact.unit.payload["concurrent_peer_action_id"] == scout.request.action_id
        assert scout.unit.payload["concurrent_peer_action_id"] == exact.request.action_id
        for intent in (exact, scout):
            package = intent.unit.payload["package"]
            package_path = root / package["path"]
            assert package_path.stat().st_size == package["bytes"]
            assert intent.unit.payload["package_manifest"]["sha256"] == (
                intent.package_manifest.manifest_sha256
            )
            assert intent.unit.payload["remote_raw_evidence_policy"] == "preserve"

    replayed = prepare_bootstrap_activation(
        repository_root=root,
        output_dir="prepared/bootstrap_activation",
        seed_receipt_paths=[prepared_repository["seed_receipt"]],
        source_epoch_receipt_paths=[prepared_repository["source_receipt"]],
        exact_source_archive_paths={
            LANE_HUBBARD: prepared_repository["exact_source_archive"]
        },
        avqds_source_archive_paths={
            LANE_HUBBARD: prepared_repository["avqds_source_archive"]
        },
        runtime_image_identity_path=prepared_repository["image_identity"],
        free_bytes=30 * 1024**3,
        recorded_at="2026-08-29T02:00:00-05:00",
    )
    assert replayed == manifest


@pytest.mark.parametrize("tamper", ("archive_bytes", "inventory_profile"))
def test_scout_preflight_reauthenticates_runtime_source_binding(
    repository: tuple[Path, Mapping[str, Any]], tamper: str
) -> None:
    root, manifest = repository
    scout_intent = load_submission_intent(
        manifest["pairs"][0]["avqds_scout"]["intent"]["path"],
        repository_root=root,
    )
    preflight_path = root / scout_intent.unit.payload["package_preflight"]["path"]
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    payload.pop("package_preflight_sha256")
    if tamper == "archive_bytes":
        payload["runtime_source_archive"]["bytes"] += 1
        expected = "runtime source byte identity diverged"
    else:
        payload["runtime_source_inventory"]["profile"] = "avqds_trajectory"
        expected = "runtime source inventory diverged"
    payload["package_preflight_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _write_json(preflight_path, payload)

    with pytest.raises(ValueError, match=expected):
        validate_avqds_prefix_package(preflight_path, repository_root=root)


def test_prepare_rejects_existing_scout_package_archive_rebinding(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    first_exact = load_submission_intent(
        manifest["pairs"][0]["exact_reference"]["intent"]["path"],
        repository_root=root,
    )
    first_scout = load_submission_intent(
        manifest["pairs"][0]["avqds_scout"]["intent"]["path"],
        repository_root=root,
    )
    original_archive = root / first_scout.unit.payload["runtime_source_archive"]["path"]
    rebound_archive = original_archive.with_name("rebound_avqds_source.tar.gz")
    shutil.copyfile(original_archive, rebound_archive)

    for pair in manifest["pairs"]:
        (root / pair["avqds_scout"]["intent"]["path"]).unlink()
    (root / "prepared/bootstrap_activation/activation_manifest.json").unlink()

    with pytest.raises(
        BootstrapActivationError,
        match="existing AVQDS scout package has another identity or runtime source archive",
    ):
        prepare_bootstrap_activation(
            repository_root=root,
            output_dir="prepared/bootstrap_activation",
            seed_receipt_paths=[
                row["path"] for row in manifest["qualified_seed_receipts"]
            ],
            source_epoch_receipt_paths=[
                row["path"] for row in manifest["source_epoch_receipts"]
            ],
            exact_source_archive_paths={
                LANE_HUBBARD: first_exact.unit.payload["runtime_source_archive"][
                    "path"
                ]
            },
            avqds_source_archive_paths={
                LANE_HUBBARD: rebound_archive.relative_to(root).as_posix()
            },
            runtime_image_identity_path=first_scout.unit.payload[
                "runtime_image_identity"
            ]["path"],
            free_bytes=30 * 1024**3,
            recorded_at="2026-08-29T02:00:00-05:00",
        )


def test_intent_first_absence_authorizes_once_then_found_recovers(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = _exact_intent(manifest)
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(root, intent_path)
    dry_run = record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )
    assert dry_run["dry_run_ad"] == {
        "path": dry_run_ad.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(dry_run_ad.read_bytes()).hexdigest(),
        "bytes": dry_run_ad.stat().st_size,
    }
    first = _snapshot(
        root,
        intent_path=intent_path,
        name="first",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
    )
    assert first.payload["reconciliation"][intent.package_manifest.job_batch_name][
        "status"
    ] == "indeterminate"
    with pytest.raises(BootstrapActivationError, match="cannot authorize"):
        authorize_from_snapshot(
            repository_root=root,
            intent_path=intent_path,
            snapshot_path=first.path,
            queue_stdout_path=first.queue_stdout,
            history_stdout_path=first.history_stdout,
        )

    later = _snapshot(
        root,
        intent_path=intent_path,
        name="later-absent",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=first.payload,
    )
    assert later.payload["reconciliation"][intent.package_manifest.job_batch_name][
        "status"
    ] == "verified_absent"
    authorization = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=later.path,
        queue_stdout_path=later.queue_stdout,
        history_stdout_path=later.history_stdout,
        prior_snapshot_path=first.path,
        prior_queue_stdout_path=first.queue_stdout,
        prior_history_stdout_path=first.history_stdout,
    )
    assert authorization["status"] == "authorized_single_submit"
    assert authorization["pre_submit_claim"]["sha256"]
    with pytest.raises(BootstrapActivationError, match="claim.*reconciliation"):
        authorize_from_snapshot(
            repository_root=root,
            intent_path=intent_path,
            snapshot_path=later.path,
            queue_stdout_path=later.queue_stdout,
            history_stdout_path=later.history_stdout,
            prior_snapshot_path=first.path,
            prior_queue_stdout_path=first.queue_stdout,
            prior_history_stdout_path=first.history_stdout,
        )

    submit = record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=authorization["authorization_path"],
        returncode=0,
        stdout="Submitting job(s).\n1 job(s) submitted to cluster 123.\n",
        submitted_at="2026-08-29T02:05:00-05:00",
    )
    assert submit["pre_submit_claim_consumption"]["sha256"]
    assert record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=authorization["authorization_path"],
        returncode=0,
        stdout="Submitting job(s).\n1 job(s) submitted to cluster 123.\n",
        submitted_at="2026-08-29T02:05:00-05:00",
    ) == submit
    with pytest.raises(BootstrapActivationError, match="different submit result"):
        record_provisional_submit_result(
            repository_root=root,
            intent_path=intent_path,
            authorization_path=authorization["authorization_path"],
            returncode=0,
            stdout="Submitting job(s).\n1 job(s) submitted to cluster 124.\n",
            submitted_at="2026-08-29T02:05:00-05:00",
        )
    found = _snapshot(
        root,
        intent_path=intent_path,
        name="later-found",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=300,
        rows=[_found_row(intent_path, root, cluster_id=123)],
        prior=later.payload,
    )
    result_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor"
        / "submission_results"
        / f"{intent.unit.unit_id}.json"
    )
    receipt = recover_submission_receipt(
        repository_root=root,
        intent_path=intent_path,
        provisional_result_path=result_path,
        later_snapshot_path=found.path,
        queue_stdout_path=found.queue_stdout,
        history_stdout_path=found.history_stdout,
        prior_snapshot_path=later.path,
        prior_queue_stdout_path=later.queue_stdout,
        prior_history_stdout_path=later.history_stdout,
    )
    assert receipt["remote"]["scheduler_identity"] == "123.0"
    assert recover_submission_receipt(
        repository_root=root,
        intent_path=intent_path,
        provisional_result_path=result_path,
        later_snapshot_path=found.path,
        queue_stdout_path=found.queue_stdout,
        history_stdout_path=found.history_stdout,
        prior_snapshot_path=later.path,
        prior_queue_stdout_path=later.queue_stdout,
        prior_history_stdout_path=later.history_stdout,
    ) == receipt
    with pytest.raises(BootstrapActivationError, match="result or receipt"):
        authorize_from_snapshot(
            repository_root=root,
            intent_path=intent_path,
            snapshot_path=later.path,
            queue_stdout_path=later.queue_stdout,
            history_stdout_path=later.history_stdout,
            prior_snapshot_path=first.path,
            prior_queue_stdout_path=first.queue_stdout,
            prior_history_stdout_path=first.history_stdout,
        )


def test_fresh_intent_in_mature_snapshot_chain_authorizes_after_its_own_absence(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    ancestor_intent_path = str(
        manifest["pairs"][1]["exact_reference"]["intent"]["path"]
    )
    intent_path = _exact_intent(manifest)
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(
        root, intent_path, name="mature-chain-dry-run"
    )
    record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )

    mature_ancestor = _snapshot(
        root,
        intent_path=ancestor_intent_path,
        name="mature-chain-ancestor",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
    )
    first_for_intent = _snapshot(
        root,
        intent_path=intent_path,
        name="mature-chain-first-for-intent",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=mature_ancestor.payload,
    )
    first_outcome = first_for_intent.payload["reconciliation"][
        intent.package_manifest.job_batch_name
    ]
    assert first_for_intent.payload["reconciliation_context"][
        "prior_snapshot"
    ] is not None
    assert first_outcome["status"] == "indeterminate"
    assert (
        first_outcome["indeterminate_reason"]
        == "first_snapshot_for_exact_intent_identity"
    )

    later = _snapshot(
        root,
        intent_path=intent_path,
        name="mature-chain-verified-absent",
        captured_at="2026-08-29T02:05:00-05:00",
        captured_at_unix_ns=300,
        prior=first_for_intent.payload,
    )
    later_outcome = later.payload["reconciliation"][
        intent.package_manifest.job_batch_name
    ]
    assert later_outcome["status"] == "verified_absent"
    assert later_outcome["absence_proof"]["prior_snapshot_sha256"] == (
        first_for_intent.payload["snapshot_sha256"]
    )

    authorization = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=later.path,
        queue_stdout_path=later.queue_stdout,
        history_stdout_path=later.history_stdout,
        prior_snapshot_path=first_for_intent.path,
        prior_queue_stdout_path=first_for_intent.queue_stdout,
        prior_history_stdout_path=first_for_intent.history_stdout,
    )
    assert authorization["status"] == "authorized_single_submit"


def test_authorization_round_trip_accepts_empty_successful_scheduler_streams(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = _exact_intent(manifest)
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(root, intent_path)
    record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )
    first = _snapshot(
        root,
        intent_path=intent_path,
        name="empty-first",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
        empty_successful_streams=True,
    )
    later = _snapshot(
        root,
        intent_path=intent_path,
        name="empty-later",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=first.payload,
        empty_successful_streams=True,
    )

    authorization = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=later.path,
        queue_stdout_path=later.queue_stdout,
        history_stdout_path=later.history_stdout,
        prior_snapshot_path=first.path,
        prior_queue_stdout_path=first.queue_stdout,
        prior_history_stdout_path=first.history_stdout,
    )

    assert authorization["status"] == "authorized_single_submit"
    assert authorization["verified_absent_snapshot"]["queue_stdout"]["bytes"] == 0
    assert authorization["verified_absent_snapshot"]["history_stdout"]["bytes"] == 0


def test_scout_receipt_recovery_authenticates_frozen_submitted_package(
    repository: tuple[Path, Mapping[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, manifest = repository
    intent_path = str(manifest["pairs"][0]["avqds_scout"]["intent"]["path"])
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(
        root, intent_path, name="frozen-scout-dry-run"
    )
    record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )
    first = _snapshot(
        root,
        intent_path=intent_path,
        name="frozen-scout-first",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
    )
    absent = _snapshot(
        root,
        intent_path=intent_path,
        name="frozen-scout-absent",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=first.payload,
    )
    authorization = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=absent.path,
        queue_stdout_path=absent.queue_stdout,
        history_stdout_path=absent.history_stdout,
        prior_snapshot_path=first.path,
        prior_queue_stdout_path=first.queue_stdout,
        prior_history_stdout_path=first.history_stdout,
    )
    record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=authorization["authorization_path"],
        returncode=0,
        stdout="Submitting job(s).\n1 job(s) submitted to cluster 456.\n",
        submitted_at="2026-08-29T02:05:00-05:00",
    )
    found = _snapshot(
        root,
        intent_path=intent_path,
        name="frozen-scout-found",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=300,
        rows=[_found_row(intent_path, root, cluster_id=456)],
        prior=absent.payload,
    )

    overlay_path = root / "pipelines/scaffold/runtime_loader.py"
    overlay_path.write_bytes(
        overlay_path.read_bytes() + b"\n# later committed runtime-loader repair\n"
    )
    preflight_path = root / intent.unit.payload["package_preflight"]["path"]
    with pytest.raises(
        AVQDSPrefixPackageError,
        match="package preflight runtime source archive is invalid",
    ) as caught:
        validate_avqds_prefix_package(preflight_path, repository_root=root)
    assert caught.value.__cause__ is not None
    assert "executor overlay bytes changed" in str(caught.value.__cause__)

    result_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor/submission_results"
        / f"{intent.unit.unit_id}.json"
    )
    assert activation_cli(
        [
            "recover",
            "--repo-root",
            str(root),
            "--intent",
            intent_path,
            "--provisional-result",
            str(result_path),
            "--later-snapshot",
            str(found.path),
            "--queue-stdout",
            str(found.queue_stdout),
            "--history-stdout",
            str(found.history_stdout),
            "--prior-snapshot",
            str(absent.path),
            "--prior-queue-stdout",
            str(absent.queue_stdout),
            "--prior-history-stdout",
            str(absent.history_stdout),
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    receipt_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor/submission_receipts"
        / f"{intent.unit.unit_id}.json"
    )
    receipt = load_submission_receipt(receipt_path, repository_root=root)
    assert summary["submission_receipt"] == receipt_path.relative_to(root).as_posix()
    assert receipt["remote"]["scheduler_identity"] == "456.0"


def test_frozen_submitted_scout_validation_rejects_package_tamper(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent = load_submission_intent(
        manifest["pairs"][0]["avqds_scout"]["intent"]["path"],
        repository_root=root,
    )
    package_path = root / intent.unit.payload["package"]["path"]
    package_path.write_bytes(package_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError):
        validate_frozen_submitted_avqds_prefix_package(
            root / intent.unit.payload["package_preflight"]["path"],
            repository_root=root,
        )


def test_empty_byte_exception_remains_explicit_and_scheduler_scoped(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    binding = {
        "path": "empty.json",
        "sha256": hashlib.sha256(b"").hexdigest(),
        "bytes": 0,
    }

    with pytest.raises(BootstrapActivationError, match="byte count changed"):
        activation_module._verified_binding(
            tmp_path,
            binding,
            label="generic evidence",
            require_bytes=True,
        )

    path, accepted = activation_module._verified_binding(
        tmp_path,
        binding,
        label="successful scheduler stdout",
        require_bytes=True,
        allow_empty_bytes=True,
    )
    assert path == empty
    assert accepted == binding


def test_found_snapshot_recovers_without_upload_dry_run_or_resubmit(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = _exact_intent(manifest)
    found = _snapshot(
        root,
        intent_path=intent_path,
        name="already-found",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
        rows=[_found_row(intent_path, root, cluster_id=321)],
    )
    outcome = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=found.path,
        queue_stdout_path=found.queue_stdout,
        history_stdout_path=found.history_stdout,
    )
    assert outcome == {
        "status": "found_recovered_without_resubmission",
        "submission_authorized": False,
        "submission_receipt": outcome["submission_receipt"],
        "cluster_id": 321,
    }


def test_found_snapshot_recovers_frozen_scout_after_overlay_evolution(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = str(manifest["pairs"][0]["avqds_scout"]["intent"]["path"])
    intent = load_submission_intent(intent_path, repository_root=root)
    found = _snapshot(
        root,
        intent_path=intent_path,
        name="already-found-frozen-scout",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
        rows=[_found_row(intent_path, root, cluster_id=322)],
    )

    overlay_path = root / "pipelines/scaffold/runtime_loader.py"
    overlay_path.write_bytes(
        overlay_path.read_bytes() + b"\n# later committed runtime-loader repair\n"
    )
    with pytest.raises(AVQDSPrefixPackageError):
        load_submission_intent(intent_path, repository_root=root)

    outcome = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=found.path,
        queue_stdout_path=found.queue_stdout,
        history_stdout_path=found.history_stdout,
    )
    assert outcome["status"] == "found_recovered_without_resubmission"
    assert outcome["submission_authorized"] is False
    assert outcome["cluster_id"] == 322
    receipt = load_submission_receipt(
        root / outcome["submission_receipt"], repository_root=root
    )
    assert receipt["remote"]["scheduler_identity"] == "322.0"


def test_absent_snapshot_requires_live_scout_validation_after_overlay_evolution(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = str(manifest["pairs"][0]["avqds_scout"]["intent"]["path"])
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(
        root, intent_path, name="stale-live-scout-dry-run"
    )
    record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )
    first = _snapshot(
        root,
        intent_path=intent_path,
        name="stale-live-scout-first",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
    )
    absent = _snapshot(
        root,
        intent_path=intent_path,
        name="stale-live-scout-absent",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=first.payload,
    )
    overlay_path = root / "pipelines/scaffold/runtime_loader.py"
    overlay_path.write_bytes(
        overlay_path.read_bytes() + b"\n# later committed runtime-loader repair\n"
    )

    with pytest.raises(AVQDSPrefixPackageError):
        authorize_from_snapshot(
            repository_root=root,
            intent_path=intent_path,
            snapshot_path=absent.path,
            queue_stdout_path=absent.queue_stdout,
            history_stdout_path=absent.history_stdout,
            prior_snapshot_path=first.path,
            prior_queue_stdout_path=first.queue_stdout,
            prior_history_stdout_path=first.history_stdout,
        )


def test_dry_run_rejects_unpreserved_digest_and_descriptor_drift(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = _exact_intent(manifest)
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(root, intent_path)
    remote_descriptor.write_bytes(b"not the final package descriptor\n")
    with pytest.raises(BootstrapActivationError, match="descriptor.*package"):
        record_dry_run_result(
            repository_root=root,
            intent_path=intent_path,
            returncode=0,
            dry_run_ad_path=dry_run_ad,
            remote_submit_descriptor_path=remote_descriptor,
            recorded_at="2026-08-29T02:02:00-05:00",
        )

    with pytest.raises(SystemExit):
        activation_cli(
            [
                "record-dry-run",
                "--repo-root",
                str(root),
                "--intent",
                intent_path,
                "--returncode",
                "0",
                "--dry-run-ad-sha256",
                "a" * 64,
            ]
        )


def test_verified_absence_requires_preserved_raw_and_prior_artifacts(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path = _exact_intent(manifest)
    intent = load_submission_intent(intent_path, repository_root=root)
    package = intent.unit.payload["package"]
    record_upload_verification(
        repository_root=root,
        intent_path=intent_path,
        remote_bytes=package["bytes"],
        remote_sha256=package["sha256"],
        recorded_at="2026-08-29T02:01:00-05:00",
    )
    dry_run_ad, remote_descriptor = _dry_run_artifacts(root, intent_path)
    record_dry_run_result(
        repository_root=root,
        intent_path=intent_path,
        returncode=0,
        dry_run_ad_path=dry_run_ad,
        remote_submit_descriptor_path=remote_descriptor,
        recorded_at="2026-08-29T02:02:00-05:00",
    )
    first = _snapshot(
        root,
        intent_path=intent_path,
        name="raw-first",
        captured_at="2026-08-29T02:03:00-05:00",
        captured_at_unix_ns=100,
    )
    later = _snapshot(
        root,
        intent_path=intent_path,
        name="raw-later",
        captured_at="2026-08-29T02:04:00-05:00",
        captured_at_unix_ns=200,
        prior=first.payload,
    )
    with pytest.raises(BootstrapActivationError, match="prior snapshot evidence"):
        authorize_from_snapshot(
            repository_root=root,
            intent_path=intent_path,
            snapshot_path=later.path,
            queue_stdout_path=later.queue_stdout,
            history_stdout_path=later.history_stdout,
        )

    later.queue_stdout.write_text("[]\n", encoding="utf-8")
    with pytest.raises(BootstrapActivationError, match="stdout digest mismatch"):
        authorize_from_snapshot(
            repository_root=root,
            intent_path=intent_path,
            snapshot_path=later.path,
            queue_stdout_path=later.queue_stdout,
            history_stdout_path=later.history_stdout,
            prior_snapshot_path=first.path,
            prior_queue_stdout_path=first.queue_stdout,
            prior_history_stdout_path=first.history_stdout,
        )


def test_unsubmitted_claim_reconciles_to_one_generation_two_authorization(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path, intent, initial_first, initial_absent, generation_one = (
        _first_generation_authorization(root, manifest, prefix="claim-recovery")
    )
    old_claim_path = root / generation_one["pre_submit_claim"]["path"]
    old_claim_bytes = old_claim_path.read_bytes()
    first_post_claim = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-post-first",
        captured_at="2026-08-29T02:05:00-05:00",
        captured_at_unix_ns=300,
    )
    later_post_claim = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-post-later",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=400,
        prior=first_post_claim.payload,
    )
    with pytest.raises(BootstrapActivationError, match="two post-claim"):
        activation_module.recover_claim_after_absence(
            repository_root=root,
            intent_path=intent_path,
            predecessor_claim_path=old_claim_path,
            first_snapshot_path=first_post_claim.path,
            first_queue_stdout_path=first_post_claim.queue_stdout,
            first_history_stdout_path=first_post_claim.history_stdout,
            later_snapshot_path=first_post_claim.path,
            later_queue_stdout_path=first_post_claim.queue_stdout,
            later_history_stdout_path=first_post_claim.history_stdout,
        )

    preclaim_later = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-preclaim-later",
        captured_at="2026-08-29T02:05:30-05:00",
        captured_at_unix_ns=350,
        prior=initial_first.payload,
    )
    with pytest.raises(BootstrapActivationError, match="strictly after.*claim"):
        activation_module.recover_claim_after_absence(
            repository_root=root,
            intent_path=intent_path,
            predecessor_claim_path=old_claim_path,
            first_snapshot_path=initial_first.path,
            first_queue_stdout_path=initial_first.queue_stdout,
            first_history_stdout_path=initial_first.history_stdout,
            later_snapshot_path=preclaim_later.path,
            later_queue_stdout_path=preclaim_later.queue_stdout,
            later_history_stdout_path=preclaim_later.history_stdout,
        )

    generation_two = activation_module.recover_claim_after_absence(
        repository_root=root,
        intent_path=intent_path,
        predecessor_claim_path=old_claim_path,
        first_snapshot_path=first_post_claim.path,
        first_queue_stdout_path=first_post_claim.queue_stdout,
        first_history_stdout_path=first_post_claim.history_stdout,
        later_snapshot_path=later_post_claim.path,
        later_queue_stdout_path=later_post_claim.queue_stdout,
        later_history_stdout_path=later_post_claim.history_stdout,
    )
    assert generation_two["claim_generation"] == 2
    assert generation_two["pre_submit_claim"]["path"] != (
        generation_one["pre_submit_claim"]["path"]
    )
    assert generation_two["claim_resolution"]["sha256"] in (
        root / generation_two["pre_submit_claim"]["path"]
    ).read_text(encoding="utf-8")
    assert old_claim_path.read_bytes() == old_claim_bytes

    with pytest.raises(BootstrapActivationError, match="resolved.*not active"):
        record_provisional_submit_result(
            repository_root=root,
            intent_path=intent_path,
            authorization_path=generation_one["authorization_path"],
            returncode=0,
            stdout="Submitting job(s).\n1 job(s) submitted to cluster 122.\n",
            submitted_at="2026-08-29T02:07:00-05:00",
        )
    submitted = record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=generation_two["authorization_path"],
        returncode=0,
        stdout="Submitting job(s).\n1 job(s) submitted to cluster 123.\n",
        submitted_at="2026-08-29T02:07:00-05:00",
    )
    assert submitted["cluster_id"] == 123
    assert submitted["claim_generation"] == 2
    assert submitted["pre_submit_claim_consumption"]["sha256"]
    assert intent.unit.unit_id in generation_two["authorization_path"]


def test_rejected_submit_reconciles_to_one_generation_two_authorization(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path, intent, _first, initial_absent, generation_one = (
        _first_generation_authorization(root, manifest, prefix="rejected-submit")
    )
    generation_one_result = record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=generation_one["authorization_path"],
        returncode=1,
        stdout="Submitting job(s)",
        stderr="ERROR: executable is unavailable\n",
        submitted_at="2026-08-29T02:05:00-05:00",
    )
    generation_one_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor"
        / "submission_results"
        / f"{intent.unit.unit_id}.json"
    )
    generation_one_bytes = generation_one_path.read_bytes()
    after_rejection = _snapshot(
        root,
        intent_path=intent_path,
        name="rejected-submit-after",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=300,
        prior=initial_absent.payload,
    )

    generation_two = activation_module.authorize_retry_after_rejected_submit(
        repository_root=root,
        intent_path=intent_path,
        failed_result_path=generation_one_path,
        later_snapshot_path=after_rejection.path,
        queue_stdout_path=after_rejection.queue_stdout,
        history_stdout_path=after_rejection.history_stdout,
        prior_snapshot_path=initial_absent.path,
        prior_queue_stdout_path=initial_absent.queue_stdout,
        prior_history_stdout_path=initial_absent.history_stdout,
    )
    assert generation_two["claim_generation"] == 2
    assert generation_two["claim_resolution"]["path"]
    resolution = activation_module.load_pre_submit_claim_resolution(
        root / generation_two["claim_resolution"]["path"],
        repository_root=root,
    )
    assert resolution["status"] == "verified_absent_after_rejected_submit"
    assert resolution["failed_submission_result"]["sha256"] == (
        generation_one_result["submission_result_sha256"]
    )
    assert generation_one_path.read_bytes() == generation_one_bytes

    submitted = record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=generation_two["authorization_path"],
        returncode=0,
        stdout="Submitting job(s).\n1 job(s) submitted to cluster 124.\n",
        submitted_at="2026-08-29T02:07:00-05:00",
    )
    assert submitted["status"] == "submitted"
    assert submitted["claim_generation"] == 2
    assert submitted["cluster_id"] == 124
    successor_name = Path(submitted["submission_authorization"]["path"]).name
    successor_path = generation_one_path.with_name(successor_name)
    assert successor_path.is_file()
    assert successor_path != generation_one_path
    assert generation_one_path.read_bytes() == generation_one_bytes

    with pytest.raises(BootstrapActivationError, match="generation-one"):
        activation_module.authorize_retry_after_rejected_submit(
            repository_root=root,
            intent_path=intent_path,
            failed_result_path=successor_path,
            later_snapshot_path=after_rejection.path,
            queue_stdout_path=after_rejection.queue_stdout,
            history_stdout_path=after_rejection.history_stdout,
            prior_snapshot_path=initial_absent.path,
            prior_queue_stdout_path=initial_absent.queue_stdout,
            prior_history_stdout_path=initial_absent.history_stdout,
        )


def test_nonzero_submit_with_acceptance_stdout_never_authorizes_retry(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path, intent, _first, initial_absent, generation_one = (
        _first_generation_authorization(
            root, manifest, prefix="contradictory-submit"
        )
    )
    contradictory = record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=generation_one["authorization_path"],
        returncode=1,
        stdout="Submitting job(s).\n1 job(s) submitted to cluster 125.\n",
        stderr="ERROR: transport reported failure after acceptance text\n",
        submitted_at="2026-08-29T02:05:00-05:00",
    )
    assert contradictory["status"] == "indeterminate"
    assert contradictory["cluster_id"] == 125
    assert contradictory["acceptance_evidence"] == (
        "condor_submit_nonzero_exit_with_cluster_id_stdout"
    )
    result_path = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor"
        / "submission_results"
        / f"{intent.unit.unit_id}.json"
    )
    after_contradiction = _snapshot(
        root,
        intent_path=intent_path,
        name="contradictory-submit-after",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=300,
        prior=initial_absent.payload,
    )

    with pytest.raises(BootstrapActivationError, match="definitive"):
        activation_module.authorize_retry_after_rejected_submit(
            repository_root=root,
            intent_path=intent_path,
            failed_result_path=result_path,
            later_snapshot_path=after_contradiction.path,
            queue_stdout_path=after_contradiction.queue_stdout,
            history_stdout_path=after_contradiction.history_stdout,
            prior_snapshot_path=initial_absent.path,
            prior_queue_stdout_path=initial_absent.queue_stdout,
            prior_history_stdout_path=initial_absent.history_stdout,
        )


def test_generation_two_consumption_blocks_false_absence_recovery(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path, intent, _initial_first, _initial_absent, generation_one = (
        _first_generation_authorization(
            root, manifest, prefix="claim-recovery-generation-two"
        )
    )
    generation_one_claim = root / generation_one["pre_submit_claim"]["path"]
    first_after_generation_one = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-generation-two-post-first",
        captured_at="2026-08-29T02:05:00-05:00",
        captured_at_unix_ns=300,
    )
    later_after_generation_one = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-generation-two-post-later",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=400,
        prior=first_after_generation_one.payload,
    )
    generation_two = activation_module.recover_claim_after_absence(
        repository_root=root,
        intent_path=intent_path,
        predecessor_claim_path=generation_one_claim,
        first_snapshot_path=first_after_generation_one.path,
        first_queue_stdout_path=first_after_generation_one.queue_stdout,
        first_history_stdout_path=first_after_generation_one.history_stdout,
        later_snapshot_path=later_after_generation_one.path,
        later_queue_stdout_path=later_after_generation_one.queue_stdout,
        later_history_stdout_path=later_after_generation_one.history_stdout,
    )
    generation_two_claim = root / generation_two["pre_submit_claim"]["path"]
    submitted = record_provisional_submit_result(
        repository_root=root,
        intent_path=intent_path,
        authorization_path=generation_two["authorization_path"],
        returncode=1,
        stdout="Submitting job(s)",
        stderr="ERROR: transport failed after claim consumption\n",
        submitted_at="2026-08-29T02:07:00-05:00",
    )
    assert submitted["claim_generation"] == 2
    assert "--g2-" in submitted["submission_authorization"]["path"]

    first_after_generation_two = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-generation-three-first",
        captured_at="2026-08-29T02:08:00-05:00",
        captured_at_unix_ns=500,
    )
    later_after_generation_two = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-recovery-generation-three-later",
        captured_at="2026-08-29T02:09:00-05:00",
        captured_at_unix_ns=600,
        prior=first_after_generation_two.payload,
    )
    with pytest.raises(
        BootstrapActivationError,
        match="submitted or recovered claim cannot be absence-resolved",
    ):
        activation_module.recover_claim_after_absence(
            repository_root=root,
            intent_path=intent_path,
            predecessor_claim_path=generation_two_claim,
            first_snapshot_path=first_after_generation_two.path,
            first_queue_stdout_path=first_after_generation_two.queue_stdout,
            first_history_stdout_path=first_after_generation_two.history_stdout,
            later_snapshot_path=later_after_generation_two.path,
            later_queue_stdout_path=later_after_generation_two.queue_stdout,
            later_history_stdout_path=later_after_generation_two.history_stdout,
        )


def test_submit_crash_claim_recovers_from_later_found_without_result(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    intent_path, intent, _first, initial_absent, authorization = (
        _first_generation_authorization(root, manifest, prefix="submit-crash")
    )
    found = _snapshot(
        root,
        intent_path=intent_path,
        name="submit-crash-found",
        captured_at="2026-08-29T02:05:00-05:00",
        captured_at_unix_ns=300,
        rows=[_found_row(intent_path, root, cluster_id=777)],
        prior=initial_absent.payload,
    )
    recovered = authorize_from_snapshot(
        repository_root=root,
        intent_path=intent_path,
        snapshot_path=found.path,
        queue_stdout_path=found.queue_stdout,
        history_stdout_path=found.history_stdout,
        prior_snapshot_path=initial_absent.path,
        prior_queue_stdout_path=initial_absent.queue_stdout,
        prior_history_stdout_path=initial_absent.history_stdout,
    )
    assert recovered["status"] == "found_recovered_without_resubmission"
    assert recovered["cluster_id"] == 777
    assert (root / authorization["pre_submit_claim"]["path"]).is_file()
    assert not (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor"
        / "submission_results"
        / f"{intent.unit.unit_id}.json"
    ).exists()


def test_claim_recovery_and_submit_consumption_commit_exclusively(
    repository: tuple[Path, Mapping[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = repository
    intent_path, intent, _initial_first, _initial_absent, generation_one = (
        _first_generation_authorization(root, manifest, prefix="claim-race")
    )
    claim_path = root / generation_one["pre_submit_claim"]["path"]
    first_post_claim = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-race-post-first",
        captured_at="2026-08-29T02:05:00-05:00",
        captured_at_unix_ns=300,
    )
    later_post_claim = _snapshot(
        root,
        intent_path=intent_path,
        name="claim-race-post-later",
        captured_at="2026-08-29T02:06:00-05:00",
        captured_at_unix_ns=400,
        prior=first_post_claim.payload,
    )
    recovery_at_publication = threading.Event()
    permit_recovery_publication = threading.Event()
    submit_started = threading.Event()
    submit_transition_reached = threading.Event()
    original_publish = activation_module._atomic_write_immutable
    original_flock = activation_module.fcntl.flock

    def controlled_publish(path: Path, payload: Mapping[str, Any]) -> str:
        if (
            threading.current_thread().name == "claim-recovery-transition"
            and path.parent.name == "pre_submit_claim_resolutions"
        ):
            recovery_at_publication.set()
            if not permit_recovery_publication.wait(timeout=10):
                raise AssertionError("test did not release claim recovery")
        status = original_publish(path, payload)
        if (
            threading.current_thread().name == "submit-consumption-transition"
            and path.parent.name == "pre_submit_claim_consumptions"
        ):
            submit_transition_reached.set()
        return status

    def controlled_flock(descriptor: int, operation: int) -> object:
        if (
            threading.current_thread().name == "submit-consumption-transition"
            and operation & activation_module.fcntl.LOCK_EX
        ):
            submit_transition_reached.set()
        return original_flock(descriptor, operation)

    monkeypatch.setattr(
        activation_module, "_atomic_write_immutable", controlled_publish
    )
    monkeypatch.setattr(activation_module.fcntl, "flock", controlled_flock)
    outcomes: dict[str, tuple[str, object]] = {}

    def run_recovery() -> None:
        try:
            value = activation_module.recover_claim_after_absence(
                repository_root=root,
                intent_path=intent_path,
                predecessor_claim_path=claim_path,
                first_snapshot_path=first_post_claim.path,
                first_queue_stdout_path=first_post_claim.queue_stdout,
                first_history_stdout_path=first_post_claim.history_stdout,
                later_snapshot_path=later_post_claim.path,
                later_queue_stdout_path=later_post_claim.queue_stdout,
                later_history_stdout_path=later_post_claim.history_stdout,
            )
        except Exception as exc:  # pragma: no branch - either outcome is asserted
            outcomes["recovery"] = ("error", exc)
        else:
            outcomes["recovery"] = ("committed", value)

    def run_submit() -> None:
        submit_started.set()
        try:
            value = record_provisional_submit_result(
                repository_root=root,
                intent_path=intent_path,
                authorization_path=generation_one["authorization_path"],
                returncode=0,
                stdout="Submitting job(s).\n1 job(s) submitted to cluster 456.\n",
                submitted_at="2026-08-29T02:07:00-05:00",
            )
        except Exception as exc:  # pragma: no branch - either outcome is asserted
            outcomes["submit"] = ("error", exc)
        else:
            outcomes["submit"] = ("committed", value)

    recovery_thread = threading.Thread(
        target=run_recovery, name="claim-recovery-transition"
    )
    submit_thread = threading.Thread(
        target=run_submit, name="submit-consumption-transition"
    )
    recovery_thread.start()
    assert recovery_at_publication.wait(timeout=10)
    submit_thread.start()
    assert submit_started.wait(timeout=10)
    assert submit_transition_reached.wait(timeout=10)
    permit_recovery_publication.set()
    recovery_thread.join(timeout=10)
    submit_thread.join(timeout=10)
    assert not recovery_thread.is_alive()
    assert not submit_thread.is_alive()

    committed = {
        name for name, (status, _value) in outcomes.items() if status == "committed"
    }
    assert committed in ({"recovery"}, {"submit"}), outcomes
    consumption = (
        root
        / "output"
        / intent.lane_id
        / "acquisition_supervisor"
        / "pre_submit_claim_consumptions"
        / f"{intent.unit.unit_id}.json"
    )
    successor_authorizations = list(
        (
            root
            / "output"
            / intent.lane_id
            / "acquisition_supervisor"
            / "submission_authorizations"
        ).glob(f"{intent.unit.unit_id}--g2-*.json")
    )
    assert not (consumption.exists() and successor_authorizations)


def test_immutable_publish_never_replaces_a_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "immutable.json"

    def competing_publish(_source: object, destination: object) -> None:
        Path(destination).write_bytes(b"competitor bytes\n")
        raise FileExistsError

    monkeypatch.setattr(activation_module.os, "link", competing_publish)
    with pytest.raises(BootstrapActivationError, match="overwrite immutable"):
        activation_module._atomic_write_immutable(target, {"owner": "coordinator"})
    assert target.read_bytes() == b"competitor bytes\n"


def test_terminal_exact_failure_prepares_one_identity_preserving_successor(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    (
        predecessor_path,
        predecessor,
        result_path,
        receipt_path,
        terminal,
        archive,
    ) = _submitted_terminal_predecessor(
        root, manifest, prefix="terminal-successor"
    )

    failure = record_exact_reference_terminal_failure_evidence(
        repository_root=root,
        predecessor_intent_path=predecessor_path,
        provisional_result_path=result_path,
        submission_receipt_path=receipt_path,
        snapshot_path=terminal.path,
        queue_stdout_path=terminal.queue_stdout,
        history_stdout_path=terminal.history_stdout,
        diagnostic_archive_path=archive,
    )
    replayed_failure = record_exact_reference_terminal_failure_evidence(
        repository_root=root,
        predecessor_intent_path=predecessor_path,
        provisional_result_path=result_path,
        submission_receipt_path=receipt_path,
        snapshot_path=terminal.path,
        queue_stdout_path=terminal.queue_stdout,
        history_stdout_path=terminal.history_stdout,
        diagnostic_archive_path=archive,
    )
    assert replayed_failure == failure
    assert failure["scientific_computation_started"] is False
    assert failure["automatic_retry_authorized"] is False

    predecessor_package = PreparedExactReferenceActionPackage.load(
        root / predecessor.unit.payload["package_preflight"]["path"],
        repository_root=root,
    )
    repaired = materialize_exact_reference_action_package(
        repository_root=root,
        output_dir=(
            "prepared/terminal_successor/"
            f"{predecessor.request.digest[:16]}/package_g2"
        ),
        request_path=root / predecessor_package.request.path,
        work_order_path=root / predecessor_package.work_order.path,
        source_epoch_receipt=predecessor_package.source_epoch,
        source_archive_path=root / predecessor_package.source_archive.path,
        runtime_image_identity=predecessor_package.runtime_image,
        resources=predecessor_package.resources,
        free_bytes=30 * 1024**3,
        operational_generation=2,
    )
    failure_path = (
        root
        / "output"
        / predecessor.lane_id
        / "acquisition_supervisor/terminal_failure_evidence"
        / f"{predecessor.unit.unit_id}.json"
    )
    successor = prepare_exact_reference_terminal_failure_successor(
        repository_root=root,
        predecessor_intent_path=predecessor_path,
        terminal_failure_evidence_path=failure_path,
        repaired_package_preflight_path=root / repaired.preflight_receipt.path,
        remote_root="paper_ii_terminal_failure_recovery",
        recorded_at="2026-08-29T02:07:00-05:00",
    )
    replayed_successor = prepare_exact_reference_terminal_failure_successor(
        repository_root=root,
        predecessor_intent_path=predecessor_path,
        terminal_failure_evidence_path=failure_path,
        repaired_package_preflight_path=root / repaired.preflight_receipt.path,
        remote_root="paper_ii_terminal_failure_recovery",
        recorded_at="2026-08-29T02:07:00-05:00",
    )

    assert replayed_successor == successor
    assert successor.request == predecessor.request
    assert successor.source_epoch == predecessor.source_epoch
    assert successor.unit.unit_id == f"{predecessor.unit.unit_id}--g2"
    assert successor.package_manifest.job_batch_name.endswith("-g2")
    assert (
        successor.package_manifest.job_batch_name
        != predecessor.package_manifest.job_batch_name
    )
    assert successor.unit.payload["operational_generation"] == 2
    assert successor.unit.payload["scientific_identity_changed"] is False
    assert successor.unit.payload["automatic_retry_authorized"] is False
    assert (
        successor.unit.payload["package"]["sha256"]
        != predecessor.unit.payload["package"]["sha256"]
    )

    archive.write_bytes(archive.read_bytes() + b"stale")
    with pytest.raises(BootstrapActivationError, match="digest changed"):
        load_exact_reference_terminal_failure_evidence(
            failure_path, repository_root=root
        )


def test_terminal_avqds_failure_prepares_one_canary_successor(
    repository: tuple[Path, Mapping[str, Any]],
) -> None:
    root, manifest = repository
    predecessor_path = _avqds_intent(manifest)
    predecessor = activation_module.load_frozen_submitted_intent(
        predecessor_path, repository_root=root
    )
    request_path = root / predecessor.unit.payload["execution_request"]["path"]
    old_order_path = request_path.parent / "work_order.json"
    old_order, source, seed = load_frozen_avqds_work_order(
        old_order_path, repository_root=root
    )

    overlay_path = root / "pipelines/time_dynamics/prospective_avqds_scout_worker.py"
    overlay_path.write_bytes(
        overlay_path.read_bytes() + b"\n# generation-two validator repair fixture\n"
    )
    _git(root, "add", overlay_path.relative_to(root).as_posix())
    _git(root, "commit", "-qm", "repair AVQDS validator")
    repaired_order = build_avqds_work_order(
        predecessor.request,
        source_epoch_receipt=source,
        seed_receipt=seed,
        repository_root=root,
    )
    assert repaired_order.request == old_order.request
    assert (
        repaired_order.executor_overlay.overlay_revision
        != old_order.executor_overlay.overlay_revision
    )
    repaired_order_path = root / "prepared/avqds_terminal_successor/work_order.json"
    write_avqds_work_order(repaired_order_path, repaired_order)
    repaired = materialize_avqds_prefix_package(
        work_order_path=repaired_order_path,
        runtime_source_archive_path=(
            "prepared/avqds_terminal_successor/runtime_source.tar.gz"
        ),
        runtime_image_identity_path=(
            root / predecessor.unit.payload["runtime_image_identity"]["path"]
        ),
        output_dir="prepared/avqds_terminal_successor/package_g2",
        repository_root=root,
        resources=AVQDSCHTCResources(),
        free_bytes=30 * 1024**3,
        operational_generation=2,
    )

    diagnostic_archive = root / "failure_diagnostics/avqds-validator.tar.gz"
    diagnostic_archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(diagnostic_archive, "w:gz") as handle:
        content = b"post-run contract validation failed\n"
        info = tarfile.TarInfo("worker_stderr.log")
        info.size = len(content)
        info.mode = 0o600
        info.mtime = 0
        handle.addfile(info, io.BytesIO(content))

    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    failure_core = {
        "schema": "paper_ii_avqds_bootstrap_terminal_failure_evidence_v1",
        "status": (
            "TERMINAL_POST_RUN_VALIDATION_FAILURE_RETAINED_NO_SCIENTIFIC_OUTPUT"
        ),
        "lane_id": predecessor.lane_id,
        "action_id": predecessor.request.action_id,
        "execution_request_sha256": predecessor.request.digest,
        "source_epoch_identity_sha256": (
            predecessor.source_epoch.source_epoch.identity_sha256
        ),
        "execution_request": {
            "path": request_path.relative_to(root).as_posix(),
            "sha256": file_sha(request_path),
        },
        "work_order": {
            "path": old_order_path.relative_to(root).as_posix(),
            "sha256": file_sha(old_order_path),
            "work_order_sha256": old_order.work_order_sha256,
            "executor_overlay_revision": old_order.executor_overlay.overlay_revision,
        },
        "archive": {
            "path": diagnostic_archive.relative_to(root).as_posix(),
            "sha256": file_sha(diagnostic_archive),
            "bytes": diagnostic_archive.stat().st_size,
            "gzip_integrity_passed": True,
            "tar_integrity_passed": True,
            "remote_raw_evidence_preserved": True,
        },
        "failure": {
            "classification": "post_run_contract_validation_failure",
            "failure_phase": (
                "post_execution_raw_run_validation_before_atomic_promotion"
            ),
            "non_resource_failure": True,
            "resource_limit_exceeded": False,
            "trajectory_accuracy_admitted": False,
            "usable_scientific_output_present": False,
        },
        "diagnosis": {
            "root_cause_confirmed": True,
            "scientific_contract_change_required": False,
        },
        "retry_disposition": {
            "automatic_retry_authorized": False,
            "peer_avqds_submissions_authorized": False,
            "recommended_canary_only": True,
            "recommended_resources": {
                "cpus": 1,
                "memory_mib": 16 * 1024,
                "disk_kib": 8 * 1024 * 1024,
                "walltime_seconds": 43_200,
            },
            "requires_digest_verified_regression_fix": True,
            "requires_new_executor_overlay_and_package_epoch": True,
            "scientific_policy_change_authorized": False,
        },
        "observed_execution": {
            "requested_cpus": 1,
            "requested_memory_mib": 16 * 1024,
            "requested_disk_kib": 8 * 1024 * 1024,
            "scientific_computation_completed": True,
        },
        "scheduler": {"job_status": 4, "exit_code": 1},
        "remote_preservation": {
            "remote_mutation_performed": False,
            "remote_raw_evidence_preserved": True,
        },
        "raw_run_recovery": {
            "recovery_status": "irrecoverable_after_condor_scratch_reclamation",
            "failure_evidence_package_included_part": False,
        },
        "submission": {
            "intent": {
                "path": predecessor.path,
                "file_sha256": predecessor.file_sha256,
                "intent_sha256": predecessor.intent_sha256,
            }
        },
    }
    failure = {
        **failure_core,
        "failure_receipt_sha256": activation_module._digest(failure_core),
    }
    failure_path = root / "failure_diagnostics/avqds_failure_receipt.json"
    failure_path.write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert (
        load_avqds_scout_terminal_failure_evidence(
            failure_path, repository_root=root
        )["failure_receipt_sha256"]
        == failure["failure_receipt_sha256"]
    )

    successor = prepare_avqds_scout_terminal_failure_successor(
        repository_root=root,
        predecessor_intent_path=predecessor_path,
        terminal_failure_evidence_path=failure_path,
        repaired_package_preflight_path=root / repaired.preflight.path,
        remote_root="paper_ii_avqds_terminal_failure_recovery",
        recorded_at="2026-08-29T03:00:00-05:00",
    )
    replayed = prepare_avqds_scout_terminal_failure_successor(
        repository_root=root,
        predecessor_intent_path=predecessor_path,
        terminal_failure_evidence_path=failure_path,
        repaired_package_preflight_path=root / repaired.preflight.path,
        remote_root="paper_ii_avqds_terminal_failure_recovery",
        recorded_at="2026-08-29T03:00:00-05:00",
    )
    assert replayed == successor
    assert successor.request == predecessor.request
    assert successor.source_epoch == predecessor.source_epoch
    assert successor.unit.unit_id == avqds_scout_submission_unit_id(
        predecessor.request, operational_generation=2
    )
    assert successor.package_manifest.job_batch_name.endswith("-g2")
    assert successor.unit.payload["operational_generation"] == 2
    assert successor.unit.payload["canary_only"] is True
    assert successor.unit.payload["scientific_identity_changed"] is False
    assert successor.unit.payload["automatic_retry_authorized"] is False
    assert successor.unit.payload["remote_workdir"].endswith("/avqds_scout_g2")
    assert (
        successor.unit.payload["package"]["sha256"]
        != predecessor.unit.payload["package"]["sha256"]
    )


@pytest.mark.parametrize(
    ("variant", "history_rows", "message"),
    (
        (
            "running",
            lambda intent_path, root: [
                _terminal_row(
                    intent_path,
                    root,
                    cluster_id=777,
                    job_status=2,
                    exit_code=1,
                )
            ],
            "terminal nonzero",
        ),
        (
            "successful",
            lambda intent_path, root: [
                _terminal_row(
                    intent_path,
                    root,
                    cluster_id=777,
                    job_status=4,
                    exit_code=0,
                )
            ],
            "terminal nonzero",
        ),
        (
            "unknown",
            lambda intent_path, root: [
                {
                    **_terminal_row(intent_path, root, cluster_id=777),
                    "PaperIIActionId": "unknown-exact-action",
                }
            ],
            "crosses unit/lane/action",
        ),
        (
            "duplicate",
            lambda intent_path, root: [
                _terminal_row(intent_path, root, cluster_id=777),
                _terminal_row(intent_path, root, cluster_id=778),
            ],
            "duplicate",
        ),
    ),
)
def test_terminal_exact_failure_rejects_nonterminal_success_and_unknown(
    repository: tuple[Path, Mapping[str, Any]],
    variant: str,
    history_rows: object,
    message: str,
) -> None:
    root, manifest = repository
    (
        predecessor_path,
        predecessor,
        result_path,
        receipt_path,
        _terminal,
        archive,
    ) = _submitted_terminal_predecessor(
        root, manifest, prefix=f"terminal-reject-{variant}"
    )
    rows = history_rows(predecessor_path, root)
    invalid = _snapshot(
        root,
        intent_path=predecessor_path,
        name=f"terminal-reject-{variant}-invalid",
        captured_at="2026-08-29T02:07:00-05:00",
        captured_at_unix_ns=400,
        history_rows=rows,
    )

    with pytest.raises(BootstrapActivationError, match=message):
        record_exact_reference_terminal_failure_evidence(
            repository_root=root,
            predecessor_intent_path=predecessor_path,
            provisional_result_path=result_path,
            submission_receipt_path=receipt_path,
            snapshot_path=invalid.path,
            queue_stdout_path=invalid.queue_stdout,
            history_stdout_path=invalid.history_stdout,
            diagnostic_archive_path=archive,
        )
    assert predecessor.unit.payload.get("operational_generation", 1) == 1
