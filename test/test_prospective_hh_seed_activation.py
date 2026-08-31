"""Contracts for deterministic local-only HH seed action materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    LANE_HH,
    SourceEpochIdentity,
)
from pipelines.time_dynamics.prospective_execution_bindings import (
    SOURCE_EPOCH_RECEIPT_V1,
    SOURCE_TREE_MANIFEST_V1,
    ExecutionRequest,
    ProspectiveExecutionError,
    SourceEpochReceipt,
)
from pipelines.time_dynamics.prospective_hh_seed_activation import (
    CANONICAL_HH_SEED_TOKENS,
    CANONICAL_SOURCE_REVISION,
    HHSeedActivationManifest,
    materialize_hh_seed_activation,
)
from pipelines.time_dynamics.prospective_hh_seed_worker import load_work_order
from pipelines.time_dynamics.runners.materialize_prospective_hh_seed_activation import (
    main,
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_receipt(
    root: Path, *, revision: str = CANONICAL_SOURCE_REVISION
) -> SourceEpochReceipt:
    epoch = SourceEpochIdentity(
        lane_id=LANE_HH,
        source_epoch_id=f"paper-ii-hh-product-test-{revision[:12]}",
        source_revision=revision,
        source_archive_sha256="c" * 64,
    )
    tree_sha = "d" * 64
    manifest_path = root / "source/source_tree_manifest.json"
    _write(
        manifest_path,
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
                    "path": manifest_path.relative_to(root).as_posix(),
                    "sha256": _file_sha(manifest_path),
                },
                "source_tree_sha256": tree_sha,
            },
            "source_epoch_receipt_sha256",
        ),
    )
    return SourceEpochReceipt.load(
        receipt_path, repository_root=root, expected_epoch=epoch
    )


def test_materializes_exactly_six_deterministic_packaging_inputs(
    tmp_path: Path,
) -> None:
    source = _source_receipt(tmp_path)
    output = tmp_path / "output/hh_activation"

    first = materialize_hh_seed_activation(
        repository_root=tmp_path,
        source_epoch_receipt=source.receipt_path,
        output_dir=output,
    )
    second = materialize_hh_seed_activation(
        repository_root=tmp_path,
        source_epoch_receipt=source.receipt_path,
        output_dir=output,
    )
    manifest = json.loads((tmp_path / first.manifest_path).read_text(encoding="utf-8"))

    assert first == second
    assert len(first.actions) == 6
    assert manifest["action_count"] == 6
    assert tuple(row["seed_token"] for row in manifest["actions"]) == CANONICAL_HH_SEED_TOKENS
    assert len({row["seed_id"] for row in manifest["actions"]}) == 6
    assert len({row["seed_identity_sha256"] for row in manifest["actions"]}) == 6
    assert len({row["optimizer_seed"] for row in manifest["actions"]}) == 6
    assert manifest["source_epoch"]["source_revision"] == CANONICAL_SOURCE_REVISION
    assert manifest["source_epoch_receipt"] == {
        "path": source.receipt_path,
        "sha256": source.receipt_sha256,
    }
    assert manifest["status"] == "PREPARED_NOT_EXECUTED"
    assert manifest["production_submission_enabled"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["vqe_executed"] is False
    assert manifest["qiskit_compilation_requested"] is False
    assert manifest["next_boundary"] == "per_action_chtc_package_builder"

    for row in manifest["actions"]:
        request_path = tmp_path / row["execution_request"]["path"]
        order_path = tmp_path / row["seed_work_order"]["path"]
        request = ExecutionRequest.from_mapping(
            json.loads(request_path.read_text(encoding="utf-8"))
        )
        order, loaded_source = load_work_order(order_path, repository_root=tmp_path)
        assert request == order.request
        assert request.action_id == row["action_id"]
        assert order.seed.seed_id == row["seed_id"]
        assert loaded_source == source
        assert row["package_inputs"]["execution_request"] == row["execution_request"]
        assert row["package_inputs"]["seed_work_order"] == row["seed_work_order"]
        assert row["package_inputs"]["source_archive_sha256"] == "c" * 64
        assert not (tmp_path / row["artifact_path"]).exists()


def test_loader_rejects_changed_per_action_request(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path)
    result = materialize_hh_seed_activation(
        repository_root=tmp_path,
        source_epoch_receipt=source.receipt_path,
        output_dir="output/hh_activation",
    )
    request_path = tmp_path / result.actions[0]["execution_request"]["path"]
    request_path.write_bytes(request_path.read_bytes() + b"\n")

    with pytest.raises(ProspectiveExecutionError, match="request file digest"):
        HHSeedActivationManifest.load(
            tmp_path / result.manifest_path, repository_root=tmp_path
        )


def test_wrong_source_revision_fails_before_materialization(tmp_path: Path) -> None:
    source = _source_receipt(tmp_path, revision="a" * 40)

    with pytest.raises(
        ProspectiveExecutionError, match="canonical HH seed source revision"
    ):
        materialize_hh_seed_activation(
            repository_root=tmp_path,
            source_epoch_receipt=source.receipt_path,
            output_dir="output/hh_activation",
        )

    assert not (tmp_path / "output/hh_activation").exists()


def test_cli_emits_local_only_manifest_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_receipt(tmp_path)

    assert (
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--source-epoch-receipt",
                source.receipt_path,
                "--output-dir",
                "output/hh_activation",
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    loaded = HHSeedActivationManifest.load(
        tmp_path / status["manifest"], repository_root=tmp_path
    )

    assert status["status"] == "PREPARED_NOT_EXECUTED"
    assert status["action_count"] == 6
    assert status["manifest_sha256"] == loaded.manifest_sha256
    assert status["production_submission_enabled"] is False
    assert status["execution_authorized"] is False
    assert status["vqe_executed"] is False
