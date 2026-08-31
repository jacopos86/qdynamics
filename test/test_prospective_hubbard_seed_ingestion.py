"""Focused contracts for prospective Hubbard-L4 seed ingestion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    BootstrapEvidence,
    LANE_HUBBARD,
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
from pipelines.time_dynamics.prospective_hubbard_seed_ingestion import (
    HUBBARD_SEED_INCREMENT_RECEIPT_V1,
    HubbardSeedIngestionError,
    PRESERVATION_MARKER,
    ingest_hubbard_seed_delta,
)
from pipelines.time_dynamics.prospective_hubbard_seed_worker import (
    EXPECTED_LOGICAL_PARAMETER_COUNT,
    EXPECTED_SECTOR_DIMENSION,
    EXPECTED_UNIQUE_GENERATOR_COUNT,
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
        lane_id=LANE_HUBBARD,
        source_epoch_id="paper-ii-hubbard-ingestion-test",
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
    receipt_path = root / "source/source_epoch_receipt.json"
    _write(
        receipt_path,
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
    return SourceEpochReceipt.load(
        receipt_path, repository_root=root, expected_epoch=epoch
    )


def _request(source: SourceEpochReceipt):
    plan = plan_bootstrap_actions(
        build_identity_registry(),
        evidence=BootstrapEvidence(
            source_epochs={LANE_HUBBARD: source.source_epoch}
        ),
        available_executors=tuple(row.binding_id for row in executor_bindings()),
    )
    action = next(
        row
        for row in plan.ready_actions
        if row.lane_id == LANE_HUBBARD and row.kind == "build_seed"
    )
    return resolve_bootstrap_action(action)


def _computation(order, *, error: float = 4.0e-7) -> SeedComputation:
    exact = -2.0
    return SeedComputation(
        variational_energy=exact + error,
        exact_energy=exact,
        exact_state_infidelity=2.0e-9,
        runtime_payload={
            "schema_version": SEED_WORKER_SCHEMA_V1,
            "pipeline": "prospective_hubbard_seed_ingestion_test_double",
            "settings": {"problem": "hubbard"},
        },
        optimizer_receipt={
            "method": "SLSQP",
            "restarts": 4,
            "optimizer_seed": order.seed.optimizer_seed,
            "restart_summaries": [
                {"restart_index": index, "success": True}
                for index in range(1, 5)
            ],
        },
        architecture_binding={
            "policy": architecture_policy(),
            "generator_pool_sha256": "b" * 64,
            "unique_generator_count": EXPECTED_UNIQUE_GENERATOR_COUNT,
            "logical_parameter_count": EXPECTED_LOGICAL_PARAMETER_COUNT,
            "sector_dimension": EXPECTED_SECTOR_DIMENSION,
            "conditioning_search_used": False,
            "snapshot_conditioning_used": False,
            "qiskit_compilation_used": False,
        },
    )


def _raw_delta(
    root: Path,
    *,
    source: SourceEpochReceipt,
    request,
    cluster_id: int,
) -> tuple[Path, Path, Path]:
    remote = root / f"remote-{cluster_id}"
    shutil.copytree(root / "source", remote / "source")
    order = build_work_order(request, source_epoch_receipt=source)
    receipt = materialize_seed(
        order,
        repository_root=remote,
        compute=lambda current: _computation(current),
    )
    increment = (
        root
        / f"output/{LANE_HUBBARD}/attempts/cluster_{cluster_id}"
        / "increments/proc0"
    )
    raw = increment / "raw/paper_ii_seed"
    raw.mkdir(parents=True)
    shutil.copyfile(remote / receipt.artifact.path, raw / "seed.json")
    shutil.copyfile(
        remote / receipt.receipt_path, raw / "seed_qualification_receipt.json"
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
    request_path = root / "requests/hubbard/execution_request.json"
    _write(request_path, request.to_json_dict())
    return increment, raw, request_path


def _ingest(
    root: Path,
    *,
    source: SourceEpochReceipt,
    request,
    cluster_id: int,
    increment: Path,
    raw: Path,
    request_path: Path,
):
    return ingest_hubbard_seed_delta(
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


def test_ingests_hubbard_seed_once_and_replays_idempotently(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    request = _request(source)
    increment, raw, request_path = _raw_delta(
        tmp_path, source=source, request=request, cluster_id=9100
    )

    result = _ingest(
        tmp_path,
        source=source,
        request=request,
        cluster_id=9100,
        increment=increment,
        raw=raw,
        request_path=request_path,
    )
    repeated = _ingest(
        tmp_path,
        source=source,
        request=request,
        cluster_id=9100,
        increment=increment,
        raw=raw,
        request_path=request_path,
    )
    receipt = json.loads((tmp_path / result.increment_receipt_path).read_text())

    assert repeated == result
    assert receipt["schema"] == HUBBARD_SEED_INCREMENT_RECEIPT_V1
    assert receipt["lane_id"] == LANE_HUBBARD
    assert receipt["cluster_id"] == 9100
    assert receipt["process_ids"] == [0]
    assert receipt["action_id"] == request.action_id
    assert receipt["seed"]["energy_error"] <= 1.0e-6
    assert receipt["seed"]["energy_gate"] == 1.0e-6
    assert receipt["seed"]["restarts"] == 4
    assert receipt["one_seed_collection"]["status"] == "complete"
    assert receipt["raw_evidence"]["remote_raw_preserved"] is True


def test_rejects_nonexact_inventory_and_wrong_expected_identity(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    request = _request(source)
    increment, raw, request_path = _raw_delta(
        tmp_path, source=source, request=request, cluster_id=9200
    )
    (raw / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(HubbardSeedIngestionError, match="inventory"):
        _ingest(
            tmp_path,
            source=source,
            request=request,
            cluster_id=9200,
            increment=increment,
            raw=raw,
            request_path=request_path,
        )
    (raw / "unexpected.txt").unlink()
    with pytest.raises(HubbardSeedIngestionError, match="action/request/source"):
        ingest_hubbard_seed_delta(
            repository_root=tmp_path,
            raw_output_root=raw.relative_to(tmp_path),
            execution_request=request_path.relative_to(tmp_path),
            source_epoch_receipt=source.receipt_path,
            expected_action_id=request.action_id,
            expected_execution_request_sha256="0" * 64,
            expected_source_epoch_identity_sha256=source.source_epoch.identity_sha256,
            cluster_id=9200,
            process_id=0,
            increment_root=increment.relative_to(tmp_path),
        )


def test_rejects_raw_pair_with_resealed_energy_above_hard_gate(
    tmp_path: Path,
) -> None:
    source = _source_receipt(tmp_path)
    request = _request(source)
    increment, raw, request_path = _raw_delta(
        tmp_path, source=source, request=request, cluster_id=9300
    )
    artifact = json.loads((raw / "seed.json").read_text())
    artifact.pop("seed_artifact_sha256")
    artifact["variational_energy"] = artifact["exact_energy"] + 2.0e-6
    _write(raw / "seed.json", _seal(artifact, "seed_artifact_sha256"))
    receipt = json.loads((raw / "seed_qualification_receipt.json").read_text())
    receipt.pop("seed_receipt_sha256")
    receipt["artifact"]["sha256"] = _file_sha(raw / "seed.json")
    receipt["variational_energy"] = artifact["variational_energy"]
    receipt["energy_error"] = 2.0e-6
    _write(
        raw / "seed_qualification_receipt.json",
        _seal(receipt, "seed_receipt_sha256"),
    )
    stdout = json.loads((raw / "worker_stdout.json").read_text())
    stdout["seed_receipt_sha256"] = _file_sha(
        raw / "seed_qualification_receipt.json"
    )
    stdout["energy_error"] = 2.0e-6
    _write(raw / "worker_stdout.json", stdout)

    with pytest.raises(HubbardSeedIngestionError, match="raw seed artifact"):
        _ingest(
            tmp_path,
            source=source,
            request=request,
            cluster_id=9300,
            increment=increment,
            raw=raw,
            request_path=request_path,
        )


def test_rejects_duplicate_seed_in_another_cluster(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    request = _request(source)
    first = _raw_delta(tmp_path, source=source, request=request, cluster_id=9400)
    _ingest(
        tmp_path,
        source=source,
        request=request,
        cluster_id=9400,
        increment=first[0],
        raw=first[1],
        request_path=first[2],
    )
    second = _raw_delta(tmp_path, source=source, request=request, cluster_id=9401)
    with pytest.raises(HubbardSeedIngestionError, match="already covered"):
        _ingest(
            tmp_path,
            source=source,
            request=request,
            cluster_id=9401,
            increment=second[0],
            raw=second[1],
            request_path=second[2],
        )
