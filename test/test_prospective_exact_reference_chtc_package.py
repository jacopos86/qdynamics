"""Local-only package gates for prospective exact-reference actions."""

from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

import pipelines.time_dynamics.prospective_exact_reference_worker as exact_worker_module
import pipelines.time_dynamics.prospective_execution_bindings as execution_bindings_module
import pipelines.time_dynamics.runners.build_prospective_exact_reference as exact_runner_module

from pipelines.artifacts.storage_guard import GIB
from pipelines.time_dynamics.prospective_bootstrap_provenance import (
    BootstrapPackageManifest,
    write_source_epoch_contracts,
)
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_exact_reference_chtc_package import (
    ExactReferencePackageError,
    PreparedExactReferenceActionPackage,
    materialize_exact_reference_action_package,
)
from pipelines.time_dynamics.prospective_exact_reference_worker import (
    build_work_order,
    write_work_order,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    SEED_WORKER_SCHEMA_V1,
    SeedComputation,
    architecture_policy,
    build_work_order as build_seed_work_order,
    materialize_seed,
)
from pipelines.time_dynamics.prospective_seed_chtc_package import (
    SeedCHTCResources,
    load_runtime_image_identity,
    write_runtime_image_identity,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seed_computation(order) -> SeedComputation:
    return SeedComputation(
        variational_energy=-1.9999995,
        exact_energy=-2.0,
        exact_state_infidelity=2.0e-9,
        runtime_payload={
            "schema_version": SEED_WORKER_SCHEMA_V1,
            "pipeline": "prospective_seed_package_test_double",
            "settings": {"adapt_pool": "fixed_scaffold_locked"},
            "initial_state": {
                "source": "test_qualified_seed",
                "handoff_state_kind": "prepared_state",
                "nq_total": 6,
                "amplitudes_qn_to_q0": {
                    "000000": {"re": 1.0, "im": 0.0}
                },
            },
            "adapt_vqe": {"structure_locked": True},
        },
        optimizer_receipt={
            "method": "test",
            "restarts": 4,
            "optimizer_seed": order.seed.optimizer_seed,
            "restart_summaries": [
                {"restart_index": index, "success": True}
                for index in range(1, 5)
            ],
        },
        architecture_binding={
            "policy": architecture_policy(),
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
        },
    )


def _inputs(tmp_path: Path):
    repo = tmp_path / "repository"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test Author")
    for relative in (
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

    # The qualified seed's frozen source epoch predates this operational
    # executor.  Package the three digest-bound overlay files without changing
    # that scientific epoch or rerunning the seed.
    bindings_target = (
        repo / "pipelines/time_dynamics/prospective_execution_bindings.py"
    )
    worker_target = (
        repo / "pipelines/time_dynamics/prospective_exact_reference_worker.py"
    )
    runner_target = (
        repo
        / "pipelines/time_dynamics/runners/build_prospective_exact_reference.py"
    )
    shutil.copyfile(Path(execution_bindings_module.__file__), bindings_target)
    worker_target.parent.mkdir(parents=True, exist_ok=True)
    runner_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(exact_worker_module.__file__), worker_target)
    shutil.copyfile(Path(exact_runner_module.__file__), runner_target)

    source = write_source_epoch_contracts(
        repository_root=repo,
        lane_id=LANE_HH,
        source_revision=revision,
        output_dir="contracts/source_epoch",
    )
    registry = build_identity_registry()
    executors = tuple(row.binding_id for row in executor_bindings())
    seed_plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(source_epochs={LANE_HH: source.source_epoch}),
        available_executors=executors,
    )
    seed_action = next(
        row
        for row in seed_plan.ready_actions
        if row.lane_id == LANE_HH and row.kind == "build_seed"
    )
    seed_request = resolve_bootstrap_action(seed_action)
    seed_order = build_seed_work_order(
        seed_request, source_epoch_receipt=source
    )
    seed_receipt = materialize_seed(
        seed_order,
        repository_root=repo,
        compute=_seed_computation,
    )
    reference_plan = plan_bootstrap_actions(
        registry,
        evidence=BootstrapEvidence(
            seed_artifact_sha256={seed_receipt.seed_id: seed_receipt.artifact.sha256},
            source_epochs={LANE_HH: source.source_epoch},
        ),
        available_executors=executors,
    )
    action = next(
        row
        for row in reference_plan.ready_actions
        if row.lane_id == LANE_HH
        and row.kind == "build_exact_reference"
        and row.inputs["seed_id"] == seed_receipt.seed_id
    )
    request = resolve_bootstrap_action(action)
    request_path = repo / "contracts/exact_reference_request.json"
    _write(request_path, request.to_json_dict())
    order = build_work_order(
        request,
        source_epoch_receipt=source,
        seed_receipt=seed_receipt,
    )
    order_path = repo / "contracts/exact_reference_work_order.json"
    write_work_order(order_path, order)

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
    image_path = repo / "runtime/paper_ii_runtime.sif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"test SIF identity bytes\n")
    identity_path = repo / "runtime/paper_ii_runtime.identity.json"
    write_runtime_image_identity(
        identity_path, image_path=image_path, repository_root=repo
    )
    image = load_runtime_image_identity(identity_path, repository_root=repo)
    return repo, request_path, order_path, source, source_archive, image


def _materialize(tmp_path: Path):
    repo, request, order, source, archive, image = _inputs(tmp_path)
    prepared = materialize_exact_reference_action_package(
        repository_root=repo,
        output_dir="packages/exact_reference_action",
        request_path=request,
        work_order_path=order,
        source_epoch_receipt=source,
        source_archive_path=archive,
        runtime_image_identity=image,
        resources=SeedCHTCResources(2, 8, 16, 43_200),
        free_bytes=30 * GIB,
    )
    return repo, prepared


def test_materializes_one_inert_digest_bound_reference_package(
    tmp_path: Path,
) -> None:
    repo, prepared = _materialize(tmp_path)
    loaded = PreparedExactReferenceActionPackage.load(
        repo / prepared.preflight_receipt.path, repository_root=repo
    )
    replayed = materialize_exact_reference_action_package(
        repository_root=repo,
        output_dir="packages/exact_reference_action",
        request_path=repo / prepared.request.path,
        work_order_path=repo / prepared.work_order.path,
        source_epoch_receipt=prepared.source_epoch,
        source_archive_path=repo / prepared.source_archive.path,
        runtime_image_identity=prepared.runtime_image,
        resources=prepared.resources,
        free_bytes=30 * GIB,
    )

    assert loaded == prepared == replayed
    assert prepared.status == "passed_local_only_dry_run_pending"
    assert prepared.production_submission_enabled is False
    assert prepared.scheduler_submission_authorized is False
    assert prepared.condor_dry_run_executed is False
    assert prepared.remote_raw_evidence_policy == "preserve"
    descriptor = (repo / prepared.submit_descriptor.path).read_text(
        encoding="utf-8"
    )
    assert descriptor.count("queue 1") == 1
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in descriptor
    assert "+PaperIIRemoteRawPreservation = True" in descriptor
    assert "+PaperIIExactReferenceId" in descriptor
    assert "max_materialize" not in descriptor.casefold()

    manifest = BootstrapPackageManifest.load(
        repo / prepared.package_manifest.path, repository_root=repo
    )
    assert manifest.package == prepared.package
    assert manifest.request.digest == prepared.request_sha256
    with tarfile.open(repo / prepared.package.path, "r:gz") as package:
        members = package.getmembers()
        worker = package.extractfile("run_exact_reference_action.sh")
        assert worker is not None
        worker_text = worker.read().decode("utf-8")
    assert all(row.mtime == 0 and row.uid == 0 and row.gid == 0 for row in members)
    assert (
        "executor_overlay/pipelines/time_dynamics/"
        "prospective_execution_bindings.py"
    ) not in {row.name for row in members}
    assert "build_prospective_exact_reference run" in worker_text
    assert "prospective_execution_bindings.py" not in worker_text
    assert "replace_source_epoch_binding=1" not in worker_text
    assert "REMOTE_RAW_PRESERVED" in worker_text
    assert "reference_data.json" in worker_text
    assert "seed.json" in worker_text

    intent = json.loads(
        (repo / prepared.intent_template.path).read_text(encoding="utf-8")
    )
    assert intent["status"] == "template_not_recorded"
    assert intent["scheduler_contact_permitted"] is False
    assert intent["scheduler_submission_authorized"] is False
    assert intent["intent_first"]["first_snapshot_outcome"] == "indeterminate"
    assert intent["remote_raw_evidence_policy"] == "preserve"


def test_package_enforces_storage_floor_before_writing(tmp_path: Path) -> None:
    repo, request, order, source, archive, image = _inputs(tmp_path)
    with pytest.raises(ExactReferencePackageError, match="10-GiB"):
        materialize_exact_reference_action_package(
            repository_root=repo,
            output_dir="packages/exact_reference_action",
            request_path=request,
            work_order_path=order,
            source_epoch_receipt=source,
            source_archive_path=archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(2, 8, 16, 43_200),
            free_bytes=10 * GIB,
        )
    assert not (repo / "packages/exact_reference_action").exists()


def test_package_rejects_source_archive_drift(tmp_path: Path) -> None:
    repo, request, order, source, archive, image = _inputs(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ExactReferencePackageError, match="runtime source archive"):
        materialize_exact_reference_action_package(
            repository_root=repo,
            output_dir="packages/exact_reference_action",
            request_path=request,
            work_order_path=order,
            source_epoch_receipt=source,
            source_archive_path=archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(2, 8, 16, 43_200),
            free_bytes=30 * GIB,
        )


def test_binding_compatibility_overlay_does_not_relax_executor_digest_gate(
    tmp_path: Path,
) -> None:
    repo, request, order, source, archive, image = _inputs(tmp_path)
    worker = repo / "pipelines/time_dynamics/prospective_exact_reference_worker.py"
    worker.write_bytes(worker.read_bytes() + b"\n# stale worker\n")

    with pytest.raises(
        ExactReferencePackageError,
        match="executor overlay changed after work-order creation",
    ):
        materialize_exact_reference_action_package(
            repository_root=repo,
            output_dir="packages/exact_reference_action",
            request_path=request,
            work_order_path=order,
            source_epoch_receipt=source,
            source_archive_path=archive,
            runtime_image_identity=image,
            resources=SeedCHTCResources(2, 8, 16, 43_200),
            free_bytes=30 * GIB,
            operational_generation=2,
        )


def test_generation_two_carries_only_the_typed_binding_repair(
    tmp_path: Path,
) -> None:
    repo, request, order, source, archive, image = _inputs(tmp_path)
    prepared = materialize_exact_reference_action_package(
        repository_root=repo,
        output_dir="packages/exact_reference_action_g2",
        request_path=request,
        work_order_path=order,
        source_epoch_receipt=source,
        source_archive_path=archive,
        runtime_image_identity=image,
        resources=SeedCHTCResources(2, 8, 16, 43_200),
        free_bytes=30 * GIB,
        operational_generation=2,
    )

    with tarfile.open(repo / prepared.package.path, "r:gz") as package:
        bindings = package.extractfile(
            "executor_overlay/pipelines/time_dynamics/"
            "prospective_execution_bindings.py"
        )
        worker = package.extractfile("run_exact_reference_action.sh")
        assert bindings is not None
        assert worker is not None
        bindings_bytes = bindings.read()
        worker_text = worker.read().decode("utf-8")
    assert prepared.operational_generation == 2
    assert bindings_bytes == Path(execution_bindings_module.__file__).read_bytes()
    assert "prospective_execution_bindings.py" in worker_text
    assert "replace_source_epoch_binding=1" in worker_text
