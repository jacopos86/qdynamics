#!/usr/bin/env python3
"""Execute one authorized fresh 0→70 conventional Append-ADAPT row."""

from __future__ import annotations

import argparse
from dataclasses import is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from derived_protocol import (  # noqa: E402
    activate_source_root,
    build_derived_protocol,
)
from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    EXPECTED_ARTIFACT_ROLES,
    EXPECTED_EXECUTION_IDS,
    PACKAGE_ID,
    SOURCE_ARCHIVE_NAME,
    SOURCE_BUNDLE_RELATIVE_ROOT,
    TARGET_HORIZON,
    PackageContractError,
    atomic_write_json,
    canonical_json_bytes,
    digested,
    load_json_object,
    sha256_file,
    validate_execution_authorization,
    validate_package,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _safe_extract_source(destination: Path) -> None:
    package = validate_package(full_archive_scan=True)
    if package["source_archive_sha256"] is None:
        raise PackageContractError("Source archive validation did not close.")
    archive = PACKAGE_DIR / SOURCE_ARCHIVE_NAME
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise PackageContractError(
                    f"Unsafe source archive member: {member.name}"
                )
        bundle.extractall(destination, filter="data")


def _typed_summary(result: Any) -> Mapping[str, Any]:
    summary = getattr(result, "paper_i_summary", None)
    if (
        summary is None
        or not is_dataclass(summary)
        or not callable(getattr(summary, "to_dict", None))
    ):
        raise PackageContractError(
            "Append result lacks its typed Paper-I summary."
        )
    payload = summary.to_dict()
    if not isinstance(payload, Mapping):
        raise PackageContractError("Typed summary did not serialize.")
    return payload


def _write_artifacts(
    *,
    output_dir: Path,
    job: Mapping[str, Any],
    authorization: Mapping[str, Any],
    protocol: Any,
    result: Any,
) -> dict[str, Any]:
    payload = result.to_dict()
    if not isinstance(payload, Mapping):
        raise PackageContractError("Append result did not serialize.")
    result_body = _mapping(
        payload.get("result_payload"), label="Append result payload"
    )
    scientific = _mapping(
        payload.get("scientific_receipts"),
        label="Append scientific receipts",
    )
    summary = _typed_summary(result)
    accepted = result_body.get("history")
    ledger = _mapping(
        result_body.get("estimator_call_ledger"),
        label="Append estimator ledger",
    )
    replay = _mapping(
        result_body.get("controller_replay_evidence"),
        label="Append replay evidence",
    )
    stop = result_body.get("stop_reason")
    if (
        int(result_body.get("controller_rounds_completed", -1))
        != TARGET_HORIZON
        or not isinstance(accepted, list)
        or len(accepted) != TARGET_HORIZON
        or stop != "maximum_controller_rounds"
        or int(summary.get("controller_rounds_completed", -1))
        != TARGET_HORIZON
        or int(summary.get("protocol_horizon", -1))
        != TARGET_HORIZON
        or summary.get("protocol_sha256") != protocol.sha256
    ):
        raise PackageContractError(
            "Fresh round-70 result/summary horizon closure failed."
        )
    checkpoint = digested(
        {
            "schema": (
                "paper_i_append_adapt_reconstruction_checkpoint_v1"
            ),
            "continuation_boundary": (
                "authenticated_reconstruction_only_v1"
            ),
            "public_resume_execution_supported": False,
            "reconstruction_fields_complete": True,
            "fresh_start_execution": True,
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "execution_id": job["execution_id"],
            "protocol_sha256": protocol.sha256,
            "controller_rounds_completed": TARGET_HORIZON,
            "accepted_operator_labels": result_body[
                "accepted_operator_labels"
            ],
            "accepted_generator_identities": result_body[
                "accepted_generator_identities"
            ],
            "logical_theta": result_body["logical_theta"],
            "controller_replay_evidence": replay,
            "controller_replay_evidence_sha256": scientific[
                "controller_replay_evidence_sha256"
            ],
            "result_payload_sha256": summary[
                "source_result_payload_sha256"
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output_dir / "result.json", payload)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "checkpoint.json", checkpoint)
    atomic_write_json(output_dir / "estimator_ledger.json", ledger)
    preliminary = {
        "checkpoint": output_dir / "checkpoint.json",
        "estimator_ledger": output_dir / "estimator_ledger.json",
        "result": output_dir / "result.json",
        "summary": output_dir / "summary.json",
    }
    manifest = digested(
        {
            "schema": (
                "paper_i_append_adapt_stationary_core_r70_"
                "execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "source_execution_id": job["source_execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_protocol_sha256": job["source_protocol"][
                "canonical_sha256"
            ],
            "derived_protocol_sha256": protocol.sha256,
            "source_horizon": 50,
            "target_horizon": TARGET_HORIZON,
            "controller_round_origin": 0,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "output_payloads": {
                role: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for role, path in sorted(preliminary.items())
            },
        }
    )
    atomic_write_json(output_dir / "execution_manifest.json", manifest)
    artifacts = {
        **preliminary,
        "execution_manifest": output_dir / "execution_manifest.json",
    }
    if set(artifacts) != set(EXPECTED_ARTIFACT_ROLES):
        raise PackageContractError("Worker artifact-role closure failed.")
    return digested(
        {
            "schema": (
                "paper_i_append_adapt_stationary_core_r70_"
                "worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "derived_protocol_sha256": protocol.sha256,
            "fresh_start": True,
            "resume_claimed": False,
            "artifacts": [
                {
                    "role": role,
                    "path": path.name,
                    "declared_canonical_path": job["artifact_paths"][role],
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for role, path in sorted(artifacts.items())
            ],
        }
    )


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    package = validate_package(full_archive_scan=False)
    job = load_json_object(job_path, label="job")
    execution_id = str(job.get("execution_id", ""))
    if (
        execution_id not in EXPECTED_EXECUTION_IDS
        or job_path.resolve()
        != (PACKAGE_DIR / "jobs" / f"{execution_id}.json").resolve()
        or job.get("derived_protocol_sha256") is None
    ):
        raise PackageContractError("Worker job identity drifted.")
    authorization = validate_execution_authorization(
        authorization_path,
        execution_id=execution_id,
    )
    if (
        package["execution_authorized"] is not False
        or output_dir.exists()
        or output_dir.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise PackageContractError(
            "Worker destination or inert-package state is unsafe."
        )
    with tempfile.TemporaryDirectory(
        prefix=f"paper_i_append_r70_{execution_id}_"
    ) as raw:
        source_root = Path(raw)
        _safe_extract_source(source_root)
        activate_source_root(source_root)
        protocol, problem, delta = build_derived_protocol(
            job=job,
            source_root=source_root,
            validate_entire_bundle=True,
        )
        if (
            protocol.sha256 != job["derived_protocol_sha256"]
            or delta["normalized_non_horizon_settings_match"] is not True
            or delta["fresh_start"] is not True
            or delta["resume_claimed"] is not False
        ):
            raise PackageContractError(
                "Runtime round-70 protocol derivation drifted."
            )
        from pipelines.static_adapt.ra_adapt import run_append_adapt

        original = Path.cwd()
        os.chdir(source_root / SOURCE_BUNDLE_RELATIVE_ROOT)
        try:
            result = run_append_adapt(problem, protocol)
        finally:
            os.chdir(original)
        receipt = _write_artifacts(
            output_dir=output_dir,
            job=job,
            authorization=authorization,
            protocol=protocol,
            result=result,
        )
    atomic_write_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = run_cell(
            job_path=args.job.resolve(),
            authorization_path=args.authorization.resolve(),
            output_dir=args.output_dir.resolve(),
            receipt_path=args.receipt.resolve(),
        )
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
