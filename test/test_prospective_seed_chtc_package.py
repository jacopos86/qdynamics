"""Local-only contracts for one-action prospective seed CHTC packages."""

from __future__ import annotations

import gzip
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from pipelines.artifacts.storage_guard import GIB
from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BootstrapPackageManifest,
    write_source_epoch_contracts,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    LANE_HUBBARD,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    build_work_order as build_hh_work_order,
    write_work_order as write_hh_work_order,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    build_work_order as build_hubbard_work_order,
    write_work_order as write_hubbard_work_order,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    PreparedSeedActionPackage,
    SeedCHTCResources,
    SeedPackageError,
    load_runtime_image_identity,
    materialize_seed_action_package,
    write_runtime_image_identity,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepared_inputs(tmp_path: Path, *, lane_id: str = LANE_HH):
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test Author")
    if lane_id == LANE_HH:
        lane_sources = (
            "pipelines/time_dynamics/runners/build_prospective_hh_seed.py",
            "pipelines/time_dynamics/prospective_hh_seed_worker.py",
        )
        build_order = build_hh_work_order
        write_order = write_hh_work_order
    elif lane_id == LANE_HUBBARD:
        lane_sources = (
            "pipelines/time_dynamics/runners/build_prospective_hubbard_seed.py",
            "pipelines/time_dynamics/prospective_hubbard_seed_worker.py",
            "pipelines/time_dynamics/pure_hubbard_fixed_vqe.py",
        )
        build_order = build_hubbard_work_order
        write_order = write_hubbard_work_order
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unsupported test lane: {lane_id}")
    for relative in (
        *lane_sources,
        "pipelines/time_dynamics/prospective_execution_bindings.py",
        "src/exact_runtime.py",
        "requirements.txt",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# exact committed test source\n", encoding="utf-8")
    _git(repo, "add", "pipelines", "src", "requirements.txt")
    _git(repo, "commit", "-qm", "exact source")
    revision = _git(repo, "rev-parse", "HEAD")

    source = write_source_epoch_contracts(
        repository_root=repo,
        lane_id=lane_id,
        source_revision=revision,
        output_dir="contracts/source_epoch",
    )
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(source_epochs={lane_id: source.source_epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.lane_id == lane_id and row.kind == "build_seed"
    )
    request = resolve_bootstrap_action(action)
    request_path = repo / "contracts/request.json"
    _write(request_path, request.to_json_dict())
    order = build_order(request, source_epoch_receipt=source)
    work_order_path = repo / "contracts/seed_work_order.json"
    write_order(work_order_path, order)

    source_archive = repo / "contracts/runtime_source.tar.gz"
    source_tar = subprocess.run(
        (
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            revision,
            "--",
            "pipelines",
            "src",
            "requirements.txt",
        ),
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    compressed = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0
    ) as handle:
        handle.write(source_tar)
    source_archive.write_bytes(compressed.getvalue())
    image = repo / "runtime/paper_ii_runtime.sif"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"test SIF identity bytes\n")
    identity_path = repo / "runtime/paper_ii_runtime.identity.json"
    write_runtime_image_identity(
        identity_path,
        image_path=image,
        repository_root=repo,
    )
    identity = load_runtime_image_identity(identity_path, repository_root=repo)
    return (
        repo,
        request_path,
        work_order_path,
        source,
        source_archive,
        identity,
    )


def _materialize(tmp_path: Path):
    repo, request, work_order, source, archive, image = _prepared_inputs(tmp_path)
    prepared = materialize_seed_action_package(
        repository_root=repo,
        output_dir="packages/seed_action",
        request_path=request,
        work_order_path=work_order,
        source_epoch_receipt=source,
        source_archive_path=archive,
        runtime_image_identity=image,
        resources=SeedCHTCResources(
            request_cpus=4,
            request_memory_gib=32,
            request_disk_gib=40,
            max_runtime_seconds=43_200,
        ),
        free_bytes=30 * GIB,
    )
    return repo, prepared


def test_materializer_emits_deterministic_one_action_local_only_package(
    tmp_path: Path,
) -> None:
    repo, prepared = _materialize(tmp_path)
    repeated = PreparedSeedActionPackage.load(
        repo / prepared.preflight_receipt.path,
        repository_root=repo,
    )
    idempotent = materialize_seed_action_package(
        repository_root=repo,
        output_dir="packages/seed_action",
        request_path=repo / prepared.request.path,
        work_order_path=repo / prepared.work_order.path,
        source_epoch_receipt=prepared.source_epoch,
        source_archive_path=repo / prepared.source_archive.path,
        runtime_image_identity=prepared.runtime_image,
        resources=prepared.resources,
        free_bytes=30 * GIB,
    )

    assert repeated == prepared == idempotent
    assert prepared.status == "passed_local_only_dry_run_pending"
    assert prepared.action_count == 1
    assert prepared.production_submission_enabled is False
    assert prepared.scheduler_submission_authorized is False
    assert prepared.condor_dry_run_executed is False
    assert prepared.remote_raw_evidence_policy == "preserve"
    assert prepared.qiskit_compilation_requested is False
    assert prepared.storage_assessment.allowed is True

    descriptor = (repo / prepared.submit_descriptor.path).read_text(
        encoding="utf-8"
    )
    assert descriptor.count("queue 1") == 1
    assert "max_materialize" not in descriptor.casefold()
    assert "queue action" not in descriptor.casefold()
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in descriptor
    assert "stream_output = False" in descriptor
    assert "stream_error = False" in descriptor
    assert "+PaperIIRemoteRawPreservation = True" in descriptor

    manifest = BootstrapPackageManifest.load(
        repo / prepared.package_manifest.path,
        repository_root=repo,
    )
    assert manifest.package == prepared.package
    assert manifest.request.digest == prepared.request_sha256
    with tarfile.open(repo / prepared.package.path, "r:gz") as archive:
        members = archive.getmembers()
        assert "input/source.tar" not in {row.name for row in members}
    assert all(member.mtime == 0 for member in members)
    assert all(member.uid == 0 and member.gid == 0 for member in members)


def test_materializer_routes_hubbard_package_to_hubbard_worker(
    tmp_path: Path,
) -> None:
    repo, request, work_order, source, archive, image = _prepared_inputs(
        tmp_path, lane_id=LANE_HUBBARD
    )
    prepared = materialize_seed_action_package(
        repository_root=repo,
        output_dir="packages/hubbard_seed_action",
        request_path=request,
        work_order_path=work_order,
        source_epoch_receipt=source,
        source_archive_path=archive,
        runtime_image_identity=image,
        resources=SeedCHTCResources(4, 24, 40, 86_400),
        free_bytes=30 * GIB,
    )

    loaded = PreparedSeedActionPackage.load(
        repo / prepared.preflight_receipt.path,
        repository_root=repo,
    )
    assert loaded == prepared
    assert prepared.source_epoch.lane_id == LANE_HUBBARD
    descriptor = (repo / prepared.submit_descriptor.path).read_text(
        encoding="utf-8"
    )
    assert f'+PaperIILaneId = "{LANE_HUBBARD}"' in descriptor
    with tarfile.open(repo / prepared.package.path, "r:gz") as package:
        worker = package.extractfile("run_seed_action.sh")
        assert worker is not None
        worker_text = worker.read().decode("utf-8")
    assert "runners.build_prospective_hubbard_seed run" in worker_text
    assert "runners.build_prospective_hh_seed run" not in worker_text


def test_materializer_rejects_source_archive_not_bound_to_epoch(
    tmp_path: Path,
) -> None:
    repo, request, work_order, source, archive, image = _prepared_inputs(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(
        SeedPackageError, match="runtime source archive.*(invalid|exact committed)"
    ):
        materialize_seed_action_package(
            repository_root=repo,
            output_dir="packages/seed_action",
            request_path=request,
            work_order_path=work_order,
            source_epoch_receipt=source,
            source_archive_path=archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(4, 32, 40, 43_200),
            free_bytes=30 * GIB,
        )


def test_materializer_rejects_runtime_image_changed_after_identity(
    tmp_path: Path,
) -> None:
    repo, request, work_order, source, archive, image = _prepared_inputs(tmp_path)
    (repo / image.path).write_bytes(b"changed image\n")

    with pytest.raises(SeedPackageError, match="runtime image digest"):
        materialize_seed_action_package(
            repository_root=repo,
            output_dir="packages/seed_action",
            request_path=request,
            work_order_path=work_order,
            source_epoch_receipt=source,
            source_archive_path=archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(4, 32, 40, 43_200),
            free_bytes=30 * GIB,
        )


def test_materializer_enforces_ten_gib_free_space_floor(tmp_path: Path) -> None:
    repo, request, work_order, source, archive, image = _prepared_inputs(tmp_path)

    with pytest.raises(SeedPackageError, match="10-GiB free-space floor"):
        materialize_seed_action_package(
            repository_root=repo,
            output_dir="packages/seed_action",
            request_path=request,
            work_order_path=work_order,
            source_epoch_receipt=source,
            source_archive_path=archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(4, 32, 40, 43_200),
            free_bytes=10 * GIB,
        )
    assert not (repo / "packages/seed_action").exists()


def test_validation_rejects_descriptor_tampering(tmp_path: Path) -> None:
    repo, prepared = _materialize(tmp_path)
    descriptor = repo / prepared.submit_descriptor.path
    descriptor.write_text(
        descriptor.read_text(encoding="utf-8").replace("queue 1", "queue 2"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        PreparedSeedActionPackage.load(
            repo / prepared.preflight_receipt.path,
            repository_root=repo,
        )


def test_intent_template_is_inert_and_package_bound(tmp_path: Path) -> None:
    repo, prepared = _materialize(tmp_path)
    template = json.loads(
        (repo / prepared.intent_template.path).read_text(encoding="utf-8")
    )

    assert template["status"] == "template_not_recorded"
    assert template["submission_intent_recorded"] is False
    assert template["scheduler_contact_permitted"] is False
    assert template["scheduler_submission_authorized"] is False
    assert template["package"]["sha256"] == prepared.package.sha256
    assert template["one_action_per_submission"] is True
    assert template["intent_first"]["first_snapshot_outcome"] == "indeterminate"
    assert template["remote_raw_evidence_policy"] == "preserve"
