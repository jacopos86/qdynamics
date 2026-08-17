#!/usr/bin/env python3
"""Validate the inert local Page-12 round-70 continuation package."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVATION_SCHEMA,
    AUTHORIZATION_SCHEMA,
    BASE_PROTOCOL_ROOT,
    BUNDLE_ID,
    BUNDLE_MANIFEST_SCHEMA,
    CAMPAIGN_ID,
    CELL_SPECS,
    CONTROLLER_AFTER_SHA256,
    CONTROLLER_BEFORE_SHA256,
    CONTROLLER_RELATIVE_PATH,
    CONTROLLER_REPAIR_ID,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    ROUTE_CONTRACT_SHA256,
    RESUME_AFTER_SHA256,
    RESUME_BEFORE_SHA256,
    RESUME_RELATIVE_PATH,
    RESUME_REPAIR_ID,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    V1_PACKAGE_MANIFEST_FILE_SHA256,
    V1_PACKAGE_MANIFEST_SHA256,
    V1_PACKAGE_RELATIVE,
    PackageContractError,
    canonical_json_bytes,
    digested,
    execution_id,
    expected_execution_ids,
    load_json,
    safe_relative_path,
    sha256_file,
    source_execution_id,
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


def _verify_binding(
    raw: Any, *, label: str, canonical: bool = False
) -> dict[str, Any] | None:
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
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} canonical digest drifted.")
    return payload


def _request_without_horizon(protocol: Mapping[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(dict(_mapping(protocol.get("request"), label="request")))
    execution = _mapping(request.get("execution"), label="execution")
    stop = _mapping(execution.get("stop"), label="stop")
    stop.pop("maximum_controller_rounds", None)
    return request


def _validate_operational_overlay_binding(
    overlay: Mapping[str, Any],
    *,
    repair_id: str,
    relative_path: str,
    before_sha256: str,
    after_sha256: str,
) -> None:
    after = _mapping(overlay.get("after"), label="controller overlay")
    expected = Path("source_overlay") / relative_path
    if safe_relative_path(
        after.get("path"), label="operational overlay path"
    ) != expected or (
        overlay.get("repair_id") != repair_id
        or overlay.get("path") != relative_path
        or overlay.get("before_sha256") != before_sha256
        or after.get("sha256") != after_sha256
        or overlay.get("scientific_protocol_changed") is not False
        or overlay.get("scientific_settings_changed") != []
    ):
        raise PackageContractError("Operational overlay binding drifted.")
    _verify_binding(after, label=f"{repair_id} overlay")


def validate(*, full_resume: bool, worker_preflight: bool) -> dict[str, Any]:
    if (PACKAGE_DIR / "submit.sub").exists() or (PACKAGE_DIR / "queue.tsv").exists():
        raise PackageContractError("Local package contains a submit/queue artifact.")
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="manifest")
    verify_self_digest(manifest, label="manifest")
    expected = list(expected_execution_ids())
    forbidden = {
        "remote_image",
        "remote_image_path",
        "remote_image_sha256",
        "remote_output_root",
        "submit_descriptor",
        "queue",
    }
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status")
        != "passed_inert_three_authenticated_continuations"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("execution_target") != "local_mac_serial"
        or manifest.get("row_count") != 3
        or manifest.get("execution_ids") != expected
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("max_concurrency") != 1
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
        or manifest.get("inherited_v1_continuation_package")
        != {
            "path": V1_PACKAGE_RELATIVE.as_posix(),
            "manifest_file_sha256": V1_PACKAGE_MANIFEST_FILE_SHA256,
            "manifest_sha256": V1_PACKAGE_MANIFEST_SHA256,
            "resume_archive_byte_identity_required": True,
        }
        or forbidden.intersection(manifest)
    ):
        raise PackageContractError("Local package identity/state drifted.")

    for row in _sequence(manifest.get("control_files"), label="controls"):
        _verify_binding(row, label="control")
    composition = _verify_binding(
        manifest.get("runtime_source_composition"),
        label="source composition",
        canonical=True,
    )
    bundle = _verify_binding(
        manifest.get("bundle_manifest"), label="bundle", canonical=True
    )
    locks = _verify_binding(
        manifest.get("source_locks"), label="source locks", canonical=True
    )
    assert composition is not None and bundle is not None and locks is not None
    overlays = _sequence(
        composition.get("operational_overlays"), label="operational overlays"
    )
    if len(overlays) != 2:
        raise PackageContractError("Operational overlay closure drifted.")
    controller_overlay = _mapping(overlays[0], label="controller overlay")
    resume_overlay = _mapping(overlays[1], label="resume overlay")
    _validate_operational_overlay_binding(
        controller_overlay,
        repair_id=CONTROLLER_REPAIR_ID,
        relative_path=CONTROLLER_RELATIVE_PATH,
        before_sha256=CONTROLLER_BEFORE_SHA256,
        after_sha256=CONTROLLER_AFTER_SHA256,
    )
    _validate_operational_overlay_binding(
        resume_overlay,
        repair_id=RESUME_REPAIR_ID,
        relative_path=RESUME_RELATIVE_PATH,
        before_sha256=RESUME_BEFORE_SHA256,
        after_sha256=RESUME_AFTER_SHA256,
    )
    if (
        bundle.get("schema") != BUNDLE_MANIFEST_SCHEMA
        or bundle.get("only_scientific_change")
        != {
            "path": "request.execution.stop.maximum_controller_rounds",
            "before": SOURCE_HORIZON,
            "after": TARGET_HORIZON,
        }
        or bundle.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or bundle.get("submission_authorized") is not False
        or controller_overlay.get("semantic_scope")
        != "accepted_energy_roundoff_only"
        or controller_overlay.get("all_non_energy_fields_exact") is not True
        or resume_overlay.get("semantic_scope")
        != "authenticated_phase0_to_phase1_resume_closure_only"
        or resume_overlay.get(
            "phase0_full_population_authentication_preserved"
        )
        is not True
        or resume_overlay.get(
            "phase1_full_population_authentication_preserved"
        )
        is not True
        or resume_overlay.get("actual_page12_weak_snapshot_hydration_passed")
        is not True
        or resume_overlay.get("actual_snapshot_route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or (PACKAGE_DIR / "source_overlay" / CONTROLLER_RELATIVE_PATH).is_symlink()
        or (PACKAGE_DIR / "source_overlay" / RESUME_RELATIVE_PATH).is_symlink()
    ):
        raise PackageContractError("Source/bundle continuation scope drifted.")

    protocols = {
        str(row["execution_id"]): row
        for row in _sequence(manifest.get("protocols"), label="protocols")
    }
    jobs = {
        str(row["execution_id"]): row
        for row in _sequence(manifest.get("jobs"), label="jobs")
    }
    resume_rows = {
        str(row["regime_id"]): row
        for row in _sequence(manifest.get("resume_inputs"), label="resume inputs")
    }
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        target_execution_id = execution_id(regime)
        if target_execution_id not in protocols or target_execution_id not in jobs:
            raise PackageContractError(f"Missing target row: {regime}")
        target = _verify_binding(
            protocols[target_execution_id],
            label=f"{regime} protocol",
            canonical=True,
        )
        job = _verify_binding(
            jobs[target_execution_id], label=f"{regime} job", canonical=True
        )
        assert target is not None and job is not None
        source_path = (
            REPO_ROOT
            / BASE_PROTOCOL_ROOT
            / f"{source_execution_id(regime)}.json"
        )
        source = load_json(source_path, label=f"{regime} source protocol")
        verify_self_digest(source, label=f"{regime} source protocol")
        if (
            target.get("horizon") != TARGET_HORIZON
            or source.get("horizon") != SOURCE_HORIZON
            or target.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or source.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or _request_without_horizon(target) != _request_without_horizon(source)
            or job.get("resume_round") != SOURCE_HORIZON
            or job.get("target_horizon") != TARGET_HORIZON
            or job.get("execution_authorized") is not False
            or job.get("submission_authorized") is not False
        ):
            raise PackageContractError(f"More than horizon changed: {regime}")
        resume_row = resume_rows[regime]
        resume = _verify_binding(
            resume_row["manifest"], label=f"{regime} resume", canonical=True
        )
        receipt = _verify_binding(
            resume_row["checkpoint_validation"],
            label=f"{regime} checkpoint validation",
            canonical=True,
        )
        assert resume is not None and receipt is not None
        if (
            resume.get("resume_round") != SOURCE_HORIZON
            or resume.get("target_round") != TARGET_HORIZON
            or resume.get("member_count") != 3
            or resume.get("pointer_closed") is not True
            or {row["role"] for row in resume["members"]}
            != {
                "checkpoint",
                "estimator_ledger_checkpoint",
                "verified_resume_sidecar",
            }
            or receipt.get("metadata", {}).get("history_count")
            != SOURCE_HORIZON
            or receipt.get("metadata", {}).get("strict_replay_passed") is not True
            or receipt.get("metadata", {}).get("route_contract_sha256")
            != ROUTE_CONTRACT_SHA256
            or resume.get("archive", {}).get("sha256")
            != spec["v1_resume_archive"]["sha256"]
            or resume.get("archive", {}).get("size_bytes")
            != spec["v1_resume_archive"]["size_bytes"]
            or resume.get("inherited_v1_authority")
            != receipt.get("inherited_v1_authority")
            or resume.get("inherited_v1_authority", {}).get("package")
            != {
                "path": V1_PACKAGE_RELATIVE.as_posix(),
                "manifest_file_sha256": V1_PACKAGE_MANIFEST_FILE_SHA256,
                "manifest_sha256": V1_PACKAGE_MANIFEST_SHA256,
            }
            or resume.get("inherited_v1_authority", {}).get(
                "archive_byte_identity_preserved"
            )
            is not True
        ):
            raise PackageContractError(f"Resume authority drifted: {regime}")
        if full_resume:
            validate_resume_archive(
                PACKAGE_DIR / resume["archive"]["path"],
                resume,
                expected_round=SOURCE_HORIZON,
                checkpoint_validation=receipt,
            )

    activation = load_json(
        PACKAGE_DIR / "activation/activation_manifest.json", label="activation"
    )
    verify_self_digest(activation, label="activation")
    request = _verify_binding(
        activation.get("activation_request"),
        label="activation request",
        canonical=True,
    )
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("status")
        != "passed_local_activation_prepared_no_execution"
        or activation.get("execution_target") != "local_mac_serial"
        or activation.get("max_concurrency") != 1
        or activation.get("authorization_count") != 3
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("launch_ready") is not False
        or activation.get("submitted") is not False
        or request is None
        or request.get("submission_authorized") is not False
    ):
        raise PackageContractError("Local activation state drifted.")
    for row in activation["authorizations"]:
        authority = _verify_binding(row, label="authorization", canonical=True)
        if (
            authority is None
            or authority.get("schema") != AUTHORIZATION_SCHEMA
            or authority.get("execution_target") != "local_mac_serial"
            or authority.get("execution_authorized") is not True
            or authority.get("submission_authorized") is not False
        ):
            raise PackageContractError("Local authorization drifted.")

    preflights: list[dict[str, Any]] = []
    if worker_preflight:
        import run_cell

        for row in jobs.values():
            preflights.append(
                run_cell.preflight(
                    PACKAGE_DIR / row["path"], verify_resume_bytes=False
                )
            )
    return digested(
        {
            "schema": "paper_i_page12_strong_r70_local_validation_v2",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "row_count": 3,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "full_resume_validation": full_resume,
            "worker_preflight_count": len(preflights),
            "max_concurrency": 1,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
            "scientific_execution_performed": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-resume", action="store_true")
    parser.add_argument("--worker-preflight", action="store_true")
    args = parser.parse_args()
    try:
        payload = validate(
            full_resume=args.full_resume,
            worker_preflight=args.worker_preflight,
        )
    except (OSError, PackageContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
