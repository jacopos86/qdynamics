#!/usr/bin/env python3
"""Validate the sealed three-row Page-10 round-70 continuation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVATION_SCHEMA,
    AUTHORIZATION_SCHEMA,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CELL_SPECS,
    CONTROLLER_AFTER_SHA256,
    CONTROLLER_RELATIVE_PATH,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REMOTE_OUTPUT_ROOT,
    RESOURCE_ENVELOPE,
    ROUTE_CONTRACT_SHA256,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    VENDORED_STREAMING_JSON_BACKEND,
    VENDORED_STREAMING_JSON_FILES,
    VENDORED_STREAMING_JSON_VERSION,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    safe_relative_path,
    sha256_file,
    validate_resume_archive,
    verify_self_digest,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _verify_binding(raw: Any, *, label: str, canonical: bool = False) -> dict[str, Any] | None:
    binding = _mapping(raw, label=f"{label} binding")
    relative = safe_relative_path(binding.get("path"), label=f"{label} path")
    path = PACKAGE_DIR / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} bytes drifted.")
    if not canonical:
        return None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical digest drifted.")
    return payload


def validate(*, full_resume: bool, worker_preflight: bool) -> dict[str, Any]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    manifest = load_json(manifest_path, label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    expected_ids = list(expected_execution_ids())
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_three_authenticated_continuations"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 3
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
    ):
        raise PackageContractError("Package identity/state drifted.")

    for raw in _sequence(manifest.get("control_files"), label="control files"):
        _verify_binding(raw, label="control file")
    bundle = _verify_binding(
        manifest.get("bundle_manifest"), label="bundle manifest", canonical=True
    )
    locks = _verify_binding(
        manifest.get("source_locks"), label="source locks", canonical=True
    )
    composition = _verify_binding(
        manifest.get("runtime_source_composition"),
        label="runtime source composition",
        canonical=True,
    )
    source_map = _verify_binding(
        manifest.get("visible_source_map"),
        label="visible source map",
        canonical=True,
    )
    assert bundle is not None and locks is not None
    assert composition is not None and source_map is not None
    overlay = _mapping(
        composition.get("operational_overlay"), label="operational overlay"
    )
    streaming_json = _mapping(
        composition.get("streaming_json_runtime"),
        label="streaming JSON runtime",
    )
    _verify_binding(overlay.get("after"), label="controller overlay")
    if (
        bundle.get("cell_count") != 3
        or bundle.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or bundle.get("source_locks_sha256") != locks.get("sha256")
        or overlay.get("path") != CONTROLLER_RELATIVE_PATH
        or overlay.get("after", {}).get("sha256") != CONTROLLER_AFTER_SHA256
        or overlay.get("scientific_protocol_changed") is not False
        or overlay.get("scientific_settings_changed") != []
        or streaming_json.get("distribution") != "ijson"
        or streaming_json.get("version") != VENDORED_STREAMING_JSON_VERSION
        or streaming_json.get("backend") != VENDORED_STREAMING_JSON_BACKEND
        or streaming_json.get("implementation")
        != "pure_python_source_locked_v1"
        or streaming_json.get("ambient_dependency_allowed") is not False
        or len(streaming_json.get("files", []))
        != len(VENDORED_STREAMING_JSON_FILES)
        or source_map.get("figure_label") != "Page 10"
        or set(source_map.get("regimes", {}))
        != {str(spec["regime_id"]) for spec in CELL_SPECS}
    ):
        raise PackageContractError("Source/settings closure drifted.")
    _verify_binding(streaming_json.get("license"), label="vendored license")
    for raw, expected_path in zip(
        _sequence(streaming_json.get("files"), label="vendored parser files"),
        VENDORED_STREAMING_JSON_FILES,
        strict=True,
    ):
        if not isinstance(raw, Mapping) or raw.get("path") != expected_path:
            raise PackageContractError("Vendored parser file ordering drifted.")
        _verify_binding(raw, label="vendored parser file")
    for raw in _sequence(
        manifest.get("visible_source_resolver_traces"), label="resolver traces"
    ):
        _verify_binding(raw, label="visible settings resolver trace")

    jobs: dict[str, dict[str, Any]] = {}
    for raw in _sequence(manifest.get("jobs"), label="job bindings"):
        job = _verify_binding(raw, label="job", canonical=True)
        assert job is not None
        execution = str(job.get("execution_id", ""))
        if (
            execution in jobs
            or execution not in expected_ids
            or job.get("resources") != RESOURCE_ENVELOPE
            or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
            or job.get("source_horizon") != SOURCE_HORIZON
            or job.get("target_horizon") != TARGET_HORIZON
        ):
            raise PackageContractError("Job matrix drifted.")
        jobs[execution] = job
    if list(jobs) != expected_ids:
        raise PackageContractError("Job ordering drifted.")
    protocols = _sequence(manifest.get("protocols"), label="protocols")
    if [row.get("execution_id") for row in protocols if isinstance(row, Mapping)] != expected_ids:
        raise PackageContractError("Protocol ordering drifted.")
    for raw in protocols:
        protocol = _verify_binding(raw, label="protocol", canonical=True)
        assert protocol is not None
        if (
            protocol.get("horizon") != TARGET_HORIZON
            or protocol.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or protocol.get("request", {}).get("execution", {}).get("stop", {}).get(
                "maximum_controller_rounds"
            )
            != TARGET_HORIZON
        ):
            raise PackageContractError("Derived protocol drifted.")

    resume_rows = _sequence(manifest.get("resume_inputs"), label="resume inputs")
    if len(resume_rows) != 3:
        raise PackageContractError("Resume matrix cardinality drifted.")
    for raw, spec in zip(resume_rows, CELL_SPECS, strict=True):
        row = _mapping(raw, label="resume row")
        resume = _verify_binding(
            row.get("manifest"), label="resume manifest", canonical=True
        )
        checkpoint_validation = _verify_binding(
            row.get("checkpoint_validation"),
            label="checkpoint validation receipt",
            canonical=True,
        )
        _verify_binding(row.get("archive"), label="resume archive")
        assert resume is not None
        assert checkpoint_validation is not None
        if (
            row.get("regime_id") != spec["regime_id"]
            or resume.get("resume_round") != spec["resume_round"]
            or resume.get("target_round") != TARGET_HORIZON
            or resume.get("archive") != row.get("archive")
            or resume.get("checkpoint_validation")
            != row.get("checkpoint_validation")
            or checkpoint_validation.get("source_validation")
            != spec["v1_checkpoint_validation"]
            or checkpoint_validation.get("archive") != row.get("archive")
            or checkpoint_validation.get("members") != resume.get("members")
            or checkpoint_validation.get("validation_authority")
            != "inherited_v1_full_stream_validation_exact_bytes_v1"
            or checkpoint_validation.get("worker_validation_scope")
            != "stream_authenticate_all_three_members_then_strict_resume_replay_v1"
            or checkpoint_validation.get(
                "accepted_state_resume_semantic_replay_required"
            )
            is not True
            or checkpoint_validation.get("ambient_ijson_required") is not False
        ):
            raise PackageContractError("Resume row drifted.")
        if full_resume:
            validate_resume_archive(
                PACKAGE_DIR / str(row["archive"]["path"]),
                resume,
                expected_round=int(spec["resume_round"]),
                checkpoint_validation=checkpoint_validation,
            )

    queue_path = PACKAGE_DIR / str(manifest["queue"]["path"])
    _verify_binding(manifest.get("queue"), label="queue")
    queue_rows = [line.split("\t") for line in queue_path.read_text().splitlines()]
    if (
        len(queue_rows) != 3
        or any(len(row) != 12 for row in queue_rows)
        or [row[0] for row in queue_rows] != expected_ids
        or any(row[8:] != ["4", "32768", "61440", "259200"] for row in queue_rows)
    ):
        raise PackageContractError("Queue is not the exact three-row matrix.")

    activation = load_json(
        PACKAGE_DIR / "activation/activation_manifest.json",
        label="activation manifest",
    )
    verify_self_digest(activation, label="activation manifest")
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("authorization_count") != 3
        or activation.get("launch_ready") is not True
        or activation.get("submitted") is not False
        or activation.get("remote_stage") is not False
        or activation.get("condor_submit") is not False
    ):
        raise PackageContractError("Activation state drifted.")
    for raw, execution in zip(
        _sequence(activation.get("authorizations"), label="authorizations"),
        expected_ids,
        strict=True,
    ):
        authority = _verify_binding(raw, label="authorization", canonical=True)
        assert authority is not None
        if (
            authority.get("schema") != AUTHORIZATION_SCHEMA
            or authority.get("execution_id") != execution
            or authority.get("package_manifest_sha256") != manifest.get("sha256")
            or authority.get("execution_authorized") is not True
            or authority.get("submission_authorized") is not True
            or authority.get("paper_evidence_adoption_authorized") is not False
            or authority.get("submitted") is not False
        ):
            raise PackageContractError("Authorization overlay drifted.")

    descriptor_path = PACKAGE_DIR / "submit.sub"
    descriptor = descriptor_path.read_text(encoding="utf-8")
    forbidden_v1_runtime_seams = (
        "paper_i_page10_strong_holstein_r70_accepted_continuations_"
        "20260809_v1_chtc",
        "paper-i-page10-strong-r70-cont-v1",
        "paper_i_page10_strong_r70_continuations_20260809_v1/outputs",
    )
    if (
        "transfer_output_files = transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
        not in descriptor
        or f"={REMOTE_OUTPUT_ROOT}/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
        not in descriptor
        or "queue execution_id, job_path, protocol_path, authorization_path, resume_archive, resume_manifest, checkpoint_validation, resume_archive_sha256, request_cpus, memory_mb, disk_mb, max_runtime_seconds"
        not in descriptor
        or "periodic_release = False" not in descriptor
        or any(seam in descriptor for seam in forbidden_v1_runtime_seams)
    ):
        raise PackageContractError("Submit lifecycle/output contract drifted.")

    preflights: list[dict[str, Any]] = []
    if worker_preflight:
        import run_cell

        for execution in expected_ids:
            preflights.append(
                run_cell.preflight(
                    PACKAGE_DIR / "jobs" / f"{execution}.json",
                    verify_resume_bytes=False,
                )
            )
    return digested(
        {
            "schema": "paper_i_page10_strong_r70_validation_v1",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "row_count": 3,
            "execution_ids": expected_ids,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "resume_rounds": {
                str(spec["regime_id"]): spec["resume_round"]
                for spec in CELL_SPECS
            },
            "target_horizon": TARGET_HORIZON,
            "resource_envelope": dict(RESOURCE_ENVELOPE),
            "full_resume_bytes_verified": full_resume,
            "checkpoint_validation_receipt_count": 3,
            "checkpoint_metadata_authority": (
                "inherited_v1_full_stream_validation_exact_bytes_v1"
            ),
            "worker_resume_contract": (
                "stream_authenticate_all_three_members_then_"
                "strict_resume_replay_v1"
            ),
            "worker_preflight_count": len(preflights),
            "explicit_transfer_output_files": True,
            "unique_posix_staging_remaps": True,
            "submitted": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-resume", action="store_true")
    parser.add_argument("--worker-preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(
            full_resume=args.full_resume,
            worker_preflight=args.worker_preflight,
        )
    except (OSError, ValueError, json.JSONDecodeError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
