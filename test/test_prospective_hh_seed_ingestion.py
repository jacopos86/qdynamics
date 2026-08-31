"""Focused contracts for prospective HH seed-result ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HH,
    SourceEpochIdentity,
    build_identity_registry,
    plan_bootstrap_actions,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    SourceEpochReceipt,
    executor_bindings,
    resolve_bootstrap_action,
)
from pipelines.time_dynamics.prospective_hh_seed_ingestion import (
    HH_SEED_INCREMENT_RECEIPT_V1,
    HHSeedIngestionError,
    PRESERVATION_MARKER,
    ingest_hh_seed_delta,
    validate_hh_seed_collection,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import (
    SEED_WORKER_SCHEMA_V1,
    SeedComputation,
    architecture_policy,
    build_work_order,
    materialize_seed,
)


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _seal(payload: dict[str, object], key: str) -> dict[str, object]:
    result = dict(payload)
    result[key] = _canonical_sha(result)
    return result


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_receipt(root: Path) -> SourceEpochReceipt:
    epoch = SourceEpochIdentity(
        lane_id=LANE_HH,
        source_epoch_id="paper-ii-hh-seed-ingestion-test",
        source_revision="a" * 40,
        source_archive_sha256="c" * 64,
    )
    tree_sha = "d" * 64
    manifest = root / "source/source_tree_manifest.json"
    _write(
        manifest,
        _seal(
            {
                "schema": SOURCE_TREE_MANIFEST_V1,
                "lane_id": epoch.lane_id,
                "source_epoch_identity_sha256": epoch.identity_sha256,
                "source_revision": epoch.source_revision,
                "source_archive_sha256": epoch.source_archive_sha256,
                "source_tree_sha256": tree_sha,
                "provenance_kind": "verified_source_archive",
            },
            "source_tree_manifest_sha256",
        ),
    )
    path = root / "source/source_epoch_receipt.json"
    _write(
        path,
        _seal(
            {
                "schema": SOURCE_EPOCH_RECEIPT_V1,
                "source_epoch": epoch.to_json_dict(),
                "source_tree_manifest": {
                    "path": manifest.relative_to(root).as_posix(),
                    "sha256": _file_sha(manifest),
                },
                "source_tree_sha256": tree_sha,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    return SourceEpochReceipt.load(path, repository_root=root, expected_epoch=epoch)


def _requests(source: SourceEpochReceipt):
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(source_epochs={LANE_HH: source.source_epoch}),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    return tuple(
        resolve_bootstrap_action(row)
        for row in plan.ready_actions
        if row.lane_id == LANE_HH and row.kind == "build_seed"
    )


def _computation(order, ordinal: int) -> SeedComputation:
    exact = -2.0 - ordinal * 0.01
    error = 2.0e-7 + ordinal * 1.0e-8
    return SeedComputation(
        variational_energy=exact + error,
        exact_energy=exact,
        exact_state_infidelity=1.0e-9 + ordinal * 1.0e-10,
        runtime_payload={
            "schema_version": SEED_WORKER_SCHEMA_V1,
            "pipeline": "prospective_hh_seed_ingestion_test_double",
            "settings": {"adapt_pool": "fixed_scaffold_locked"},
            "adapt_vqe": {"structure_locked": True},
        },
        optimizer_receipt={
            "method": "L-BFGS-B",
            "restarts": 4,
            "optimizer_seed": order.seed.optimizer_seed,
            "restart_summaries": [
                {"restart_index": index, "success": True}
                for index in range(1, 5)
            ],
        },
        architecture_binding={
            "policy": architecture_policy(),
            "architecture_id": f"ingestion-test-{ordinal}",
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
        },
    )


def _raw_delta(
    root: Path,
    remote_root: Path,
    *,
    source: SourceEpochReceipt,
    request,
    cluster_id: int,
    ordinal: int,
) -> tuple[Path, Path]:
    remote_source = remote_root / "source"
    shutil.copytree(root / "source", remote_source / "source", dirs_exist_ok=True)
    order = build_work_order(request, source_epoch_receipt=source)
    receipt = materialize_seed(
        order,
        repository_root=remote_source,
        compute=lambda current: _computation(current, ordinal),
    )
    increment = (
        root
        / f"output/paper_ii_hh_product_v1/attempts/cluster_{cluster_id}"
        / "increments/proc0"
    )
    raw = increment / "raw/paper_ii_seed"
    raw.mkdir(parents=True)
    remote_artifact = remote_source / receipt.artifact.path
    shutil.copyfile(remote_artifact, raw / "seed.json")
    shutil.copyfile(
        remote_source / receipt.receipt_path,
        raw / "seed_qualification_receipt.json",
    )
    _write(
        raw / "worker_stdout.json",
        {
            "status": "qualified",
            "seed_id": receipt.seed_id,
            "seed_receipt": receipt.receipt_path,
            "seed_receipt_sha256": receipt.receipt_sha256,
            "energy_error": receipt.energy_error,
            "exact_state_infidelity": receipt.exact_state_infidelity,
        },
    )
    (raw / "worker_stderr.log").write_text("", encoding="utf-8")
    (raw / "REMOTE_RAW_PRESERVED").write_bytes(PRESERVATION_MARKER)
    request_path = root / f"requests/{ordinal}/execution_request.json"
    _write(request_path, request.to_json_dict())
    return increment, raw


def _ingest(
    root: Path,
    *,
    source: SourceEpochReceipt,
    request,
    cluster_id: int,
    increment: Path,
    raw: Path,
):
    request_paths = sorted((root / "requests").glob("*/execution_request.json"))
    request_path = next(
        path
        for path in request_paths
        if json.loads(path.read_text(encoding="utf-8"))["action_id"]
        == request.action_id
    )
    return ingest_hh_seed_delta(
        repository_root=root,
        raw_output_root=raw.relative_to(root),
        execution_request=request_path.relative_to(root),
        source_epoch_receipt=source.receipt_path,
        expected_action_id=request.action_id,
        expected_execution_request_sha256=request.digest,
        expected_source_epoch_identity_sha256=source.source_epoch.identity_sha256,
        cluster_id=cluster_id,
        process_id=0,
        increment_root=increment.relative_to(root),
    )


def test_ingests_one_seed_atomically_and_replays_idempotently(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    request = _requests(source)[0]
    increment, raw = _raw_delta(
        tmp_path,
        tmp_path / "remote",
        source=source,
        request=request,
        cluster_id=8100,
        ordinal=0,
    )

    result = _ingest(
        tmp_path,
        source=source,
        request=request,
        cluster_id=8100,
        increment=increment,
        raw=raw,
    )
    repeated = _ingest(
        tmp_path,
        source=source,
        request=request,
        cluster_id=8100,
        increment=increment,
        raw=raw,
    )
    receipt_path = tmp_path / result.increment_receipt_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    canonical_dir = (tmp_path / result.seed_receipt.artifact.path).parent

    assert repeated == result
    assert set(path.name for path in canonical_dir.iterdir()) == {
        "seed.json",
        "seed_qualification_receipt.json",
    }
    assert receipt["schema"] == HH_SEED_INCREMENT_RECEIPT_V1
    assert receipt["cluster_id"] == 8100
    assert receipt["process_ids"] == [0]
    assert receipt["action_id"] == request.action_id
    assert receipt["raw_evidence"]["remote_raw_preserved"] is True
    assert receipt["six_seed_collection"]["status"] == "pending"
    assert result.collection_status == "pending"


def test_rejects_partial_duplicate_and_divergent_evidence(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    requests = _requests(source)
    request = requests[0]
    increment, raw = _raw_delta(
        tmp_path,
        tmp_path / "remote-a",
        source=source,
        request=request,
        cluster_id=8200,
        ordinal=0,
    )
    (raw / "REMOTE_RAW_PRESERVED").unlink()
    with pytest.raises(HHSeedIngestionError, match="inventory"):
        _ingest(
            tmp_path,
            source=source,
            request=request,
            cluster_id=8200,
            increment=increment,
            raw=raw,
        )
    (raw / "REMOTE_RAW_PRESERVED").write_bytes(PRESERVATION_MARKER)
    first = _ingest(
        tmp_path,
        source=source,
        request=request,
        cluster_id=8200,
        increment=increment,
        raw=raw,
    )

    second_increment, second_raw = _raw_delta(
        tmp_path,
        tmp_path / "remote-b",
        source=source,
        request=request,
        cluster_id=8201,
        ordinal=0,
    )
    with pytest.raises(HHSeedIngestionError, match="already covered"):
        _ingest(
            tmp_path,
            source=source,
            request=request,
            cluster_id=8201,
            increment=second_increment,
            raw=second_raw,
        )

    artifact = tmp_path / first.seed_receipt.artifact.path
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(HHSeedIngestionError, match="canonical evidence"):
        _ingest(
            tmp_path,
            source=source,
            request=request,
            cluster_id=8200,
            increment=increment,
            raw=raw,
        )

    partial_request = requests[1]
    partial_increment, partial_raw = _raw_delta(
        tmp_path,
        tmp_path / "remote-c",
        source=source,
        request=partial_request,
        cluster_id=8202,
        ordinal=1,
    )
    partial_target = tmp_path / partial_request.inputs["artifact_path"]
    partial_target.parent.mkdir(parents=True)
    shutil.copyfile(partial_raw / "seed.json", partial_target)
    with pytest.raises(HHSeedIngestionError, match="partial"):
        _ingest(
            tmp_path,
            source=source,
            request=partial_request,
            cluster_id=8202,
            increment=partial_increment,
            raw=partial_raw,
        )


def test_sixth_increment_validates_complete_independent_collection(
    tmp_path: Path,
) -> None:
    source = _source_receipt(tmp_path)
    requests = _requests(source)
    assert len(requests) == 6

    results = []
    for ordinal, request in enumerate(requests):
        cluster_id = 8300 + ordinal
        increment, raw = _raw_delta(
            tmp_path,
            tmp_path / f"remote-{ordinal}",
            source=source,
            request=request,
            cluster_id=cluster_id,
            ordinal=ordinal,
        )
        results.append(
            _ingest(
                tmp_path,
                source=source,
                request=request,
                cluster_id=cluster_id,
                increment=increment,
                raw=raw,
            )
        )

    collection = validate_hh_seed_collection(
        repository_root=tmp_path,
        source_epoch_receipt=source.receipt_path,
    )
    final_receipt = json.loads(
        (tmp_path / results[-1].increment_receipt_path).read_text(encoding="utf-8")
    )

    assert len(collection) == 6
    assert len({row.artifact.sha256 for row in collection}) == 6
    assert len({row.optimizer_execution_id for row in collection}) == 6
    assert results[-1].collection_status == "complete"
    assert final_receipt["six_seed_collection"]["status"] == "complete"
    assert len(final_receipt["six_seed_collection"]["seed_receipt_sha256"]) == 6
