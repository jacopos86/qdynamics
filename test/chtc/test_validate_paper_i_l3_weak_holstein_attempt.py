from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "validate_paper_i_l3_weak_holstein_attempt_20260813.py"
)
SPEC = importlib.util.spec_from_file_location("l3_attempt_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _json_bytes(value: Any) -> bytes:
    return validator.canonical_json_bytes(value) + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _authorization(
    *,
    execution_id: str,
    job: dict[str, Any],
    manifest: dict[str, Any],
    activation_request: dict[str, Any],
    image_runtime_probe: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
            "schema": validator.AUTHORIZATION_SCHEMA,
            "package_id": validator.PACKAGE_ID,
            "campaign_id": validator.CAMPAIGN_ID,
            "bundle_id": validator.BUNDLE_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "source_archive_sha256": manifest["source_archive"]["sha256"],
            "activation_request": activation_request,
            "image_runtime_probe": image_runtime_probe,
            "pinned_image_path": validator.REMOTE_IMAGE_PATH,
            "pinned_image_sha256": validator.REMOTE_IMAGE_SHA256,
            "scope": "single_cell_chtc_execution_only",
            "authorization_kind": (
                "explicit_user_execution_and_submission_authority"
            ),
            "execution_authorized": True,
            "submission_authorized": True,
            "paper_evidence_adoption_authorized": False,
            "submitted": False,
    }
    if overrides:
        payload.update(overrides)
    return validator.digested(payload)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _binding(path: Path, *, root: Path) -> dict[str, Any]:
    payload = _load(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path.read_bytes()),
        "size_bytes": path.stat().st_size,
        "canonical_sha256": validator.canonical_sha256(
            {key: value for key, value in payload.items() if key != "sha256"}
        ),
    }


def _activation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_execution_id: str,
    manifest: dict[str, Any],
    authorization_overrides: dict[str, Any] | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "activation"
    root.mkdir()
    execution_ids = validator._queue_execution_ids()
    request_payload = {
        "schema": validator.ACTIVATION_REQUEST_SCHEMA,
        "package_id": validator.PACKAGE_ID,
        "campaign_id": validator.CAMPAIGN_ID,
        "bundle_id": validator.BUNDLE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "requested_execution_ids": list(execution_ids),
        "scope": "prepare_matched_six_cell_chtc_execution_and_submission_v1",
        "authorization_kind": (
            "explicit_user_execution_and_submission_authority"
        ),
        "explicit_user_authority_recorded": True,
        "execution_authorized": True,
        "submission_authorized": True,
        "paper_evidence_adoption_authorized": False,
        "submitted": False,
    }
    if request_overrides:
        request_payload.update(request_overrides)
    request = validator.digested(request_payload)
    request_path = root / "activation_request.json"
    _write_json(request_path, request)
    request_binding = _binding(request_path, root=root)

    probe = validator.digested(
        {
            "schema": "test_pinned_image_runtime_probe_v1",
            "status": "passed",
            "image_sha256": validator.REMOTE_IMAGE_SHA256,
            "probe": {
                "resolved_backend_name": "FakeMarrakesh",
                "backend_resolution_kind": "fake_exact",
            },
        }
    )
    image_probe = validator.digested(
        {
            "schema": "test_package_validation_v1",
            "status": "passed_inert_package",
            "package_manifest_sha256": manifest["sha256"],
            "launch_ready": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "pinned_image_runtime_probe": probe,
        }
    )
    probe_path = root / "image_runtime_probe.json"
    _write_json(probe_path, image_probe)
    probe_binding = _binding(probe_path, root=root)

    authorization_bindings: list[dict[str, Any]] = []
    target_authorization: dict[str, Any] | None = None
    target_authorization_path: Path | None = None
    for execution_id in execution_ids:
        job = _load(validator.PACKAGE_DIR / "jobs" / f"{execution_id}.json")
        authorization = _authorization(
            execution_id=execution_id,
            job=job,
            manifest=manifest,
            activation_request=request_binding,
            image_runtime_probe=probe_binding,
            overrides=(
                authorization_overrides
                if execution_id == target_execution_id
                else None
            ),
        )
        path = root / "authorizations" / f"{execution_id}.json"
        _write_json(path, authorization)
        authorization_bindings.append(
            {"execution_id": execution_id, **_binding(path, root=root)}
        )
        if execution_id == target_execution_id:
            target_authorization = authorization
            target_authorization_path = path
    assert target_authorization is not None
    assert target_authorization_path is not None

    submit_path = root / "submit.sub"
    submit_path.write_text("# inert test submit descriptor\n", encoding="utf-8")
    activation_manifest = validator.digested(
        {
            "schema": validator.ACTIVATION_MANIFEST_SCHEMA,
            "status": "passed_activation_prepared_no_submission",
            "package_id": validator.PACKAGE_ID,
            "campaign_id": validator.CAMPAIGN_ID,
            "bundle_id": validator.BUNDLE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "activation_request": request_binding,
            "image_runtime_probe": probe_binding,
            "pinned_image_path": validator.REMOTE_IMAGE_PATH,
            "pinned_image_sha256": validator.REMOTE_IMAGE_SHA256,
            "authorizations": authorization_bindings,
            "authorization_count": len(authorization_bindings),
            "submit_descriptor": {
                "path": "submit.sub",
                "sha256": _sha256(submit_path.read_bytes()),
                "size_bytes": submit_path.stat().st_size,
            },
            "package_relative_to_submit_root": "test/package",
            "activation_relative_to_submit_root": "test/activation",
            "launch_ready": True,
            "execution_authorized": True,
            "submission_authorized": True,
            "paper_evidence_adoption_authorized": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    activation_manifest_path = root / "activation_manifest.json"
    _write_json(activation_manifest_path, activation_manifest)
    monkeypatch.setattr(
        validator, "ACTIVATION_MANIFEST_SHA256", activation_manifest["sha256"]
    )
    monkeypatch.setattr(
        validator,
        "ACTIVATION_MANIFEST_FILE_SHA256",
        _sha256(activation_manifest_path.read_bytes()),
    )
    monkeypatch.setattr(
        validator, "ACTIVATION_REQUEST_SHA256", request["sha256"]
    )
    monkeypatch.setattr(
        validator,
        "ACTIVATION_REQUEST_FILE_SHA256",
        _sha256(request_path.read_bytes()),
    )
    return (
        activation_manifest_path,
        target_authorization_path,
        target_authorization,
    )


def _archive_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    worker_exit_status: int = 0,
    receipt_authorization_sha256: str | None = None,
    execution_manifest_rounds: int = 50,
    malformed_json_role: str | None = None,
    tamper_role_after_binding: str | None = None,
    omit_role: str | None = None,
    add_symlink: bool = False,
    cluster_id: int = validator.EXPECTED_CLUSTER_ID,
    execution_index: int = 0,
    corrupt_receipt_self_digest: bool = False,
    authorization_overrides: dict[str, Any] | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> tuple[Path, str, Path, Path]:
    execution_id = validator._queue_execution_ids()[execution_index]
    package = validator.PACKAGE_DIR
    manifest = _load(package / "package_manifest.json")
    job = _load(package / "jobs" / f"{execution_id}.json")
    (
        activation_manifest_path,
        authorization_path,
        authorization,
    ) = _activation_authority(
        tmp_path,
        monkeypatch,
        target_execution_id=execution_id,
        manifest=manifest,
        authorization_overrides=authorization_overrides,
        request_overrides=request_overrides,
    )

    expected_paths = {
        role: row["path"] for role, row in job["expected_run_artifacts"].items()
    }
    files: dict[str, bytes] = {}
    for role in validator.JSON_ARTIFACT_ROLES:
        files[expected_paths[role]] = (
            b"{not-json\n"
            if role == malformed_json_role
            else _json_bytes({"schema": f"test_{role}_v1", "role": role})
        )

    output_payloads = {
        role: {
            "path": expected_paths[role],
            "sha256": _sha256(files[expected_paths[role]]),
            "size_bytes": len(files[expected_paths[role]]),
        }
        for role in validator.JSON_ARTIFACT_ROLES
    }
    execution_manifest = validator.digested(
        {
            "schema": validator.EXECUTION_MANIFEST_SCHEMA,
            "status": "passed",
            "package_id": validator.PACKAGE_ID,
            "campaign_id": validator.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "execution_entrypoint": job["execution_entrypoint"],
            "target_horizon": 50,
            "controller_rounds_completed": execution_manifest_rounds,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "output_payloads": output_payloads,
        }
    )
    files[expected_paths["execution_manifest"]] = _json_bytes(
        execution_manifest
    )
    artifacts = [
        {
            "path": expected_paths[role],
            "sha256": _sha256(files[expected_paths[role]]),
            "size_bytes": len(files[expected_paths[role]]),
        }
        for role in validator.ARTIFACT_ROLES
    ]
    worker_receipt = validator.digested(
        {
            "schema": validator.WORKER_RECEIPT_SCHEMA,
            "status": "passed",
            "package_id": validator.PACKAGE_ID,
            "campaign_id": validator.CAMPAIGN_ID,
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": (
                authorization["sha256"]
                if receipt_authorization_sha256 is None
                else receipt_authorization_sha256
            ),
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "artifacts": artifacts,
        }
    )
    if corrupt_receipt_self_digest:
        worker_receipt["sha256"] = "0" * 64
    files["worker_receipt.json"] = _json_bytes(worker_receipt)
    files["worker_exit_status.txt"] = f"{worker_exit_status}\n".encode("ascii")
    if tamper_role_after_binding is not None:
        files[expected_paths[tamper_role_after_binding]] += b" "
    if omit_role is not None:
        del files[expected_paths[omit_role]]

    proc_id = execution_index
    archive_path = (
        tmp_path / f"{execution_id}__{cluster_id}__{proc_id}.tar.gz"
    )
    directories = {
        parent.as_posix()
        for name in files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for directory in sorted(directories):
            info = tarfile.TarInfo(f"./{directory}")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if add_symlink:
            link = tarfile.TarInfo("./unsafe_link")
            link.type = tarfile.SYMTYPE
            link.linkname = "worker_receipt.json"
            archive.addfile(link)
    return (
        archive_path,
        execution_id,
        activation_manifest_path,
        authorization_path,
    )


@pytest.mark.parametrize("execution_index", range(6))
def test_valid_fetched_attempt_closes_all_five_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_index: int,
) -> None:
    archive, execution_id, activation_manifest, authorization = _archive_fixture(
        tmp_path,
        monkeypatch,
        execution_index=execution_index,
    )
    receipt = validator.validate_attempt(
        archive_path=archive,
        expected_execution_id=execution_id,
        activation_manifest_path=activation_manifest,
        authorization_path=authorization,
    )
    assert receipt["status"] == "passed_validated_no_adoption"
    assert receipt["cluster_id"] == validator.EXPECTED_CLUSTER_ID
    assert receipt["proc_id"] == execution_index
    assert receipt["controller_rounds_completed"] == 50
    assert receipt["scheduler_identity_provenance"] == {
        "kind": "archive_basename_and_sealed_queue_position_v1",
        "cluster_proc_attested_inside_archive": False,
        "limitation": (
            "v3 worker receipt and execution manifest do not carry scheduler "
            "cluster/proc IDs"
        ),
    }
    assert [row["path"] for row in receipt["artifact_bindings"]] == [
        _load(validator.PACKAGE_DIR / "jobs" / f"{execution_id}.json")[
            "expected_run_artifacts"
        ][role]["path"]
        for role in validator.ARTIFACT_ROLES
    ]
    assert receipt["paper_evidence_adopted"] is False
    assert receipt["external_state_changed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"worker_exit_status": 2}, "nonzero"),
        ({"receipt_authorization_sha256": "f" * 64}, "receipt binding"),
        ({"execution_manifest_rounds": 49}, "Execution manifest binding"),
        ({"tamper_role_after_binding": "summary"}, "hash/size binding"),
        ({"malformed_json_role": "result"}, "unreadable JSON"),
        ({"omit_role": "checkpoint"}, "regular-file closure"),
        ({"add_symlink": True}, "links are forbidden"),
        ({"corrupt_receipt_self_digest": True}, "self-digest drifted"),
    ],
)
def test_attempt_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    archive, execution_id, activation_manifest, authorization = _archive_fixture(
        tmp_path, monkeypatch, **kwargs
    )
    with pytest.raises(validator.AttemptValidationError, match=message):
        validator.validate_attempt(
            archive_path=archive,
            expected_execution_id=execution_id,
            activation_manifest_path=activation_manifest,
            authorization_path=authorization,
        )


def test_attempt_rejects_wrong_cluster_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, execution_id, activation_manifest, authorization = _archive_fixture(
        tmp_path,
        monkeypatch,
        cluster_id=validator.EXPECTED_CLUSTER_ID + 1,
    )
    with pytest.raises(
        validator.AttemptValidationError,
        match="cluster/proc/execution mapping",
    ):
        validator.validate_attempt(
            archive_path=archive,
            expected_execution_id=execution_id,
            activation_manifest_path=activation_manifest,
            authorization_path=authorization,
        )


@pytest.mark.parametrize(
    "authorization_overrides",
    [
        {"bundle_id": "forged_bundle"},
        {"pinned_image_sha256": "0" * 64},
        {"activation_request": {"path": "missing.json"}},
    ],
)
def test_forged_self_digested_authorization_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorization_overrides: dict[str, Any],
) -> None:
    archive, execution_id, activation_manifest, authorization = _archive_fixture(
        tmp_path,
        monkeypatch,
        authorization_overrides=authorization_overrides,
    )
    with pytest.raises(
        validator.AttemptValidationError,
        match="Execution authorization binding drifted",
    ):
        validator.validate_attempt(
            archive_path=archive,
            expected_execution_id=execution_id,
            activation_manifest_path=activation_manifest,
            authorization_path=authorization,
        )


def test_recomputed_activation_manifest_is_not_submitted_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, execution_id, activation_manifest, authorization = _archive_fixture(
        tmp_path, monkeypatch
    )
    payload = _load(activation_manifest)
    payload["bundle_id"] = "forged_bundle"
    payload = validator.digested(payload)
    _write_json(activation_manifest, payload)
    with pytest.raises(
        validator.AttemptValidationError,
        match="exact submitted authority",
    ):
        validator.validate_attempt(
            archive_path=archive,
            expected_execution_id=execution_id,
            activation_manifest_path=activation_manifest,
            authorization_path=authorization,
        )


def test_cli_does_not_resolve_away_archive_symlink_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive, execution_id, activation_manifest, authorization = _archive_fixture(
        tmp_path, monkeypatch
    )
    payload_path = tmp_path / "payload.tar.gz"
    archive.rename(payload_path)
    archive.symlink_to(payload_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            VALIDATOR_PATH.as_posix(),
            "--archive",
            archive.as_posix(),
            "--expected-execution-id",
            execution_id,
            "--activation-manifest",
            activation_manifest.as_posix(),
            "--authorization",
            authorization.as_posix(),
        ],
    )
    assert validator.main() == 2
    assert "unavailable/unsafe" in capsys.readouterr().err


def test_cluster_specific_activation_pins_match_submission_receipt() -> None:
    receipt = _load(
        validator.SCRIPT_DIR
        / "paper_i_l3_weak_holstein_page12_append6_r50_20260812_"
        "v3_chtc_submission_receipt_9650825.json"
    )
    assert receipt["sha256"] == validator.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    assert receipt["submission"]["cluster_id"] == validator.EXPECTED_CLUSTER_ID
    assert (
        receipt["activation_manifest_sha256"]
        == validator.ACTIVATION_MANIFEST_SHA256
    )
    assert (
        receipt["activation_manifest_file_sha256"]
        == validator.ACTIVATION_MANIFEST_FILE_SHA256
    )
    assert (
        receipt["activation_request_sha256"]
        == validator.ACTIVATION_REQUEST_SHA256
    )
    assert (
        receipt["activation_request_file_sha256"]
        == validator.ACTIVATION_REQUEST_FILE_SHA256
    )
