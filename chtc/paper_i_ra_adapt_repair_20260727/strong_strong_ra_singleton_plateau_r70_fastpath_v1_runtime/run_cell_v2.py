#!/usr/bin/env python3
"""Execute one explicitly authorized retention-v2 overlay row."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import run_cell as base_runtime  # noqa: E402
from operational_overlay_v2_contract import (  # noqa: E402
    AUTHORIZATION_V2_SCHEMA,
    CHECKPOINT_MEMBER,
    COLLISION_CLEARANCE_SCHEMA,
    COLLISION_EVIDENCE_NAME,
    EFFECTIVE_EXECUTION_CONTRACT_SCHEMA,
    EFFECTIVE_SOURCES_NAME,
    EXECUTION_PLAN_V2_NAME,
    JOBS_V2_DIR,
    JOB_V2_SCHEMA,
    OVERLAY_ID,
    OVERLAY_MANIFEST_NAME,
    OVERLAY_MANIFEST_SCHEMA,
    OVERLAY_PACKAGE_ID,
    REPAIRED_CHECKPOINT_SHA256,
    SOURCE_LOCK_AUDIT_V2_NAME,
    build_effective_execution_contract,
    effective_contract_sha256,
)
from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    PACKAGE_ID,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    load_json,
    safe_relative_path,
    sha256_file,
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


def _package_path(value: Any, *, label: str) -> Path:
    relative = safe_relative_path(value, label=label)
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(
            f"{label} escaped the package."
        ) from exc
    return path


def _verify_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    path = _package_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size
        != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    observed = verify_self_digest(payload, label=label)
    if observed != binding.get("canonical_sha256"):
        raise PackageContractError(
            f"{label} canonical binding drifted."
        )
    return path, payload


def _load_v2_job(job_path: Path) -> dict[str, Any]:
    manifest = load_json(
        PACKAGE_DIR / OVERLAY_MANIFEST_NAME,
        label="overlay manifest",
    )
    verify_self_digest(manifest, label="overlay manifest")
    if (
        manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA
        or manifest.get("overlay_id") != OVERLAY_ID
        or manifest.get("package_id") != OVERLAY_PACKAGE_ID
        or manifest.get("base_package_id") != PACKAGE_ID
        or manifest.get("status")
        != "passed_inert_collision_blocked"
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("authority_overlay_present") is not False
        or manifest.get("collision_clearance_overlay_present")
        is not False
    ):
        raise PackageContractError(
            "Runtime overlay is not the bound inert v2 package."
        )
    if (
        (PACKAGE_DIR / "submit.sub").exists()
        or (PACKAGE_DIR / "authority").exists()
        or (PACKAGE_DIR / "collision_clearance").exists()
    ):
        raise PackageContractError(
            "The inert overlay gained package-local execution state."
        )
    documents: dict[str, dict[str, Any]] = {}
    for key, name, label in (
        (
            "effective_sources",
            EFFECTIVE_SOURCES_NAME,
            "effective sources",
        ),
        (
            "collision_evidence",
            COLLISION_EVIDENCE_NAME,
            "collision evidence",
        ),
        (
            "source_lock_audit",
            SOURCE_LOCK_AUDIT_V2_NAME,
            "source-lock audit",
        ),
        (
            "execution_plan",
            EXECUTION_PLAN_V2_NAME,
            "execution plan",
        ),
    ):
        binding = _mapping(
            manifest.get(key), label=f"{label} binding"
        )
        if binding.get("path") != name:
            raise PackageContractError(f"{label} path drifted.")
        _path, payload = _verify_binding(
            binding, label=label, canonical=True
        )
        assert payload is not None
        documents[key] = payload
    plan = documents["execution_plan"]
    collision = documents["collision_evidence"]
    if (
        plan.get("submission_ready") is not False
        or plan.get("execution_authorized") is not False
        or collision.get("blocking") is not True
        or collision.get("external_state_revalidation_required")
        is not True
        or collision.get("may_submit") is not False
    ):
        raise PackageContractError(
            "Overlay submission/collision gate drifted."
        )

    job = load_json(job_path, label="v2 job")
    job_digest = verify_self_digest(job, label="v2 job")
    execution_id = str(job.get("execution_id", ""))
    bindings = {
        str(row["execution_id"]): row
        for row in _sequence(
            manifest.get("jobs"), label="v2 job bindings"
        )
        if isinstance(row, Mapping)
    }
    binding = _mapping(
        bindings.get(execution_id), label="v2 job binding"
    )
    expected_path = (
        PACKAGE_DIR / JOBS_V2_DIR / f"{execution_id}.json"
    ).resolve()
    if (
        job_path.resolve() != expected_path
        or binding.get("path")
        != f"{JOBS_V2_DIR}/{execution_id}.json"
        or binding.get("canonical_sha256") != job_digest
        or binding.get("sha256") != sha256_file(job_path)
        or int(binding.get("size_bytes", -1))
        != job_path.stat().st_size
        or job.get("schema") != JOB_V2_SCHEMA
        or job.get("package_id") != OVERLAY_PACKAGE_ID
        or job.get("overlay_id") != OVERLAY_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("base_package_id") != PACKAGE_ID
        or job.get("effective_sources_sha256")
        != documents["effective_sources"]["sha256"]
        or job.get("execution_plan_sha256") != plan["sha256"]
        or job.get("source_lock_audit_sha256")
        != documents["source_lock_audit"]["sha256"]
        or job.get("collision_evidence_sha256")
        != collision["sha256"]
        or job.get("global_submission_blocked") is not True
        or job.get("submission_ready") is not False
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError(
            "V2 worker job/cross-document identity drifted."
        )
    contract = _mapping(
        job.get("effective_execution_contract"),
        label="effective execution contract",
    )
    contract_sha = effective_contract_sha256(contract)
    if (
        contract.get("schema")
        != EFFECTIVE_EXECUTION_CONTRACT_SCHEMA
        or contract_sha
        != job.get("effective_execution_contract_sha256")
        or contract.get("scientific_settings_sha256")
        != job.get("scientific_settings_sha256")
        or contract.get("operational_settings_sha256")
        != job.get("operational_settings_sha256")
        or plan.get("effective_execution_contracts", {})
        .get(execution_id, {})
        .get("sha256")
        != contract_sha
    ):
        raise PackageContractError(
            "V2 effective execution contract binding drifted."
        )
    sources = _mapping(
        documents["effective_sources"].get("families"),
        label="effective source families",
    )
    family = _mapping(
        sources.get(job.get("effective_source_family")),
        label="effective source family",
    )
    if (
        job.get("effective_source_archive")
        != family.get("effective_archive")
        or job.get("effective_source_archive_manifest")
        != family.get("effective_manifest")
        or job.get("effective_source_delta_receipt")
        != family.get("delta_receipt")
        or job.get("source_archive")
        != family.get("effective_archive")
        or job.get("source_archive_manifest")
        != family.get("effective_manifest")
        or family.get("checkpoint_member") != CHECKPOINT_MEMBER
        or family.get("effective_checkpoint_sha256")
        != REPAIRED_CHECKPOINT_SHA256
        or family.get("delta_proof", {}).get(
            "scientific_settings_changed"
        )
        != []
    ):
        raise PackageContractError(
            "V2 effective source family binding drifted."
        )
    _verify_binding(
        _mapping(
            job.get("effective_source_archive"),
            label="effective source archive",
        ),
        label="effective source archive",
    )
    _verify_binding(
        _mapping(
            job.get("effective_source_archive_manifest"),
            label="effective source manifest",
        ),
        label="effective source manifest",
        canonical=True,
    )
    _delta_path, delta = _verify_binding(
        _mapping(
            job.get("effective_source_delta_receipt"),
            label="effective source delta receipt",
        ),
        label="effective source delta receipt",
        canonical=True,
    )
    changed = _sequence(
        delta.get("changed_members"),
        label="source-delta members",
    )
    if (
        delta.get("scientific_settings_changed") != []
        or len(changed) != 1
        or changed[0].get("path") != CHECKPOINT_MEMBER
        or changed[0].get("repaired_sha256")
        != REPAIRED_CHECKPOINT_SHA256
        or changed[0].get("scientific_protocol_change")
        is not False
        or changed[0].get("controller_semantics_change")
        is not False
    ):
        raise PackageContractError(
            "Effective source delta is not observation-only."
        )
    return job


def _validate_clearance(
    path: Path, *, job: Mapping[str, Any]
) -> dict[str, Any]:
    clearance = load_json(path, label="collision clearance")
    verify_self_digest(clearance, label="collision clearance")
    collision = _mapping(
        job.get("collision"), label="job collision"
    )
    evidence = _sequence(
        clearance.get("evidence"), label="clearance evidence"
    )
    if (
        clearance.get("schema") != COLLISION_CLEARANCE_SCHEMA
        or clearance.get("overlay_id") != OVERLAY_ID
        or clearance.get("package_id") != OVERLAY_PACKAGE_ID
        or clearance.get("execution_id")
        != job.get("execution_id")
        or clearance.get("job_spec_sha256") != job.get("sha256")
        or clearance.get("cluster_id")
        != collision.get("cluster_id")
        or clearance.get("proc_id") != collision.get("proc_id")
        or clearance.get("source_execution_id")
        != collision.get("source_execution_id")
        or clearance.get("external_state_revalidated") is not True
        or clearance.get("predecessor_live") is not False
        or clearance.get("predecessor_terminal_or_retired")
        is not True
        or clearance.get("fresh_execution_authorized") is not True
        or clearance.get("predecessor_removal_authorized") is not False
        or not clearance.get("sealed_utc")
        or clearance.get("status") != "passed"
        or not evidence
    ):
        raise PackageContractError(
            "Fresh row lacks a sealed exact-predecessor clearance."
        )
    for index, row in enumerate(evidence):
        binding = _mapping(
            row, label=f"clearance evidence {index}"
        )
        evidence_path = Path(str(binding.get("path", ""))).resolve()
        if (
            not evidence_path.is_file()
            or evidence_path.is_symlink()
            or evidence_path.stat().st_size
            != int(binding.get("size_bytes", -1))
            or sha256_file(evidence_path)
            != binding.get("sha256")
        ):
            raise PackageContractError(
                "Collision-clearance evidence binding drifted."
            )
    return clearance


def _validate_authorization(
    path: Path,
    *,
    job: Mapping[str, Any],
    clearance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    authorization = load_json(
        path, label="v2 execution authorization"
    )
    verify_self_digest(
        authorization, label="v2 execution authorization"
    )
    clearance_sha = (
        clearance.get("sha256")
        if clearance is not None
        else None
    )
    if (
        authorization.get("schema") != AUTHORIZATION_V2_SCHEMA
        or authorization.get("overlay_id") != OVERLAY_ID
        or authorization.get("package_id") != OVERLAY_PACKAGE_ID
        or authorization.get("execution_id")
        != job.get("execution_id")
        or authorization.get("job_spec_sha256")
        != job.get("sha256")
        or authorization.get("effective_execution_contract_sha256")
        != job.get("effective_execution_contract_sha256")
        or authorization.get("scientific_settings_sha256")
        != job.get("scientific_settings_sha256")
        or authorization.get("operational_settings_sha256")
        != job.get("operational_settings_sha256")
        or authorization.get("effective_source_archive_sha256")
        != job.get("effective_source_archive", {}).get("sha256")
        or authorization.get("effective_source_delta_receipt_sha256")
        != job.get("effective_source_delta_receipt", {}).get(
            "canonical_sha256"
        )
        or authorization.get("collision_clearance_sha256")
        != clearance_sha
        or authorization.get("scope")
        != "single_cell_execution_only"
        or authorization.get("authorization_kind")
        != "explicit_user_execution_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get(
            "global_submission_block_acknowledged"
        )
        is not True
    ):
        raise PackageContractError(
            "V2 execution authorization is absent, stale, or out of scope."
        )
    return authorization


def _write_result_artifacts(
    *,
    staging: Path,
    job: Mapping[str, Any],
    authorization: Mapping[str, Any],
    clearance: Mapping[str, Any] | None,
    protocol: Any,
    result: Any,
    effective_contract: Mapping[str, Any],
) -> dict[str, Any]:
    result_payload = result.to_dict()
    if not isinstance(result_payload, Mapping):
        raise PackageContractError("RA result did not serialize.")
    summary_value = result.run.paper_i_summary
    summary = (
        summary_value.to_dict()
        if callable(getattr(summary_value, "to_dict", None))
        else None
    )
    rounds = len(result.run.accepted_trajectory)
    minimum = (
        50
        if job["execution_mode"]
        == "authenticated_resume_50_to_70"
        else 1
    )
    if (
        not isinstance(summary, Mapping)
        or result.protocol.sha256 != protocol.sha256
        or int(protocol.horizon) != TARGET_HORIZON
        or not minimum <= rounds <= TARGET_HORIZON
        or int(summary.get("available_controller_rounds", -1))
        != rounds
    ):
        raise PackageContractError(
            "V2 round-70 result closure failed."
        )
    base_runtime._write_json(staging / "result.json", result_payload)
    base_runtime._write_json(staging / "summary.json", summary)
    preliminary = {
        "checkpoint": staging / "checkpoint.json",
        "estimator_ledger": staging / "estimator_ledger.json",
        "result": staging / "result.json",
        "summary": staging / "summary.json",
    }
    if any(not path.is_file() for path in preliminary.values()):
        raise PackageContractError(
            "V2 continuation output artifacts are incomplete."
        )
    execution_manifest = digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_r70_"
                "continuation_execution_manifest_v2"
            ),
            "status": "passed",
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "execution_id": job["execution_id"],
            "execution_mode": job["execution_mode"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "collision_clearance_sha256": (
                clearance["sha256"]
                if clearance is not None
                else None
            ),
            "effective_execution_contract_sha256": (
                effective_contract["sha256"]
            ),
            "scientific_settings_sha256": effective_contract[
                "scientific_settings_sha256"
            ],
            "operational_settings_sha256": effective_contract[
                "operational_settings_sha256"
            ],
            "effective_source_archive_sha256": job[
                "effective_source_archive"
            ]["sha256"],
            "effective_source_manifest_sha256": job[
                "effective_source_archive_manifest"
            ]["canonical_sha256"],
            "effective_source_delta_receipt_sha256": job[
                "effective_source_delta_receipt"
            ]["canonical_sha256"],
            "effective_checkpoint_source_sha256": (
                REPAIRED_CHECKPOINT_SHA256
            ),
            "changed_source_members": [CHECKPOINT_MEMBER],
            "scientific_settings_changed_by_operational_overlay": [],
            "source_horizon": 50,
            "target_horizon": TARGET_HORIZON,
            "controller_rounds_available": rounds,
            "output_payloads": {
                role: {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for role, path in sorted(preliminary.items())
            },
        }
    )
    base_runtime._write_json(
        staging / "execution_manifest.json",
        execution_manifest,
    )
    return execution_manifest


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    collision_clearance_path: Path | None,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    job = _load_v2_job(job_path)
    fresh = job.get("execution_mode") == "fresh_0_to_70"
    if fresh:
        if collision_clearance_path is None:
            collision = _mapping(
                job.get("collision"), label="fresh collision"
            )
            raise PackageContractError(
                "Fresh 0→70 row remains blocked by predecessor "
                f"{collision.get('cluster_id')}."
                f"{collision.get('proc_id')}; a sealed external "
                "collision clearance is required."
            )
        clearance = _validate_clearance(
            collision_clearance_path, job=job
        )
    else:
        if (
            job.get("execution_mode")
            != "authenticated_resume_50_to_70"
            or job.get("resume_input") is None
            or collision_clearance_path is not None
        ):
            raise PackageContractError(
                "Resume row mode/clearance binding drifted."
            )
        clearance = None
    authorization = _validate_authorization(
        authorization_path, job=job, clearance=clearance
    )
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise PackageContractError(
            "V2 worker destination already exists."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{job['execution_id']}.v2.",
        dir=output_dir.parent,
    ) as raw:
        temporary = Path(raw)
        source_root = temporary / "source"
        resume_root = temporary / "resume"
        staging = temporary / "artifacts"
        staging.mkdir()
        base_runtime._extract_source(job, source_root)
        checkpoint = (
            None
            if fresh
            else base_runtime._extract_resume(job, resume_root)
        )
        base_runtime._activate_source_root(source_root)
        protocol, problem, _delta = base_runtime._derived_protocol(
            job=job, source_root=source_root
        )
        controls = base_runtime._observation_controls(
            output_root=staging,
            checkpoint_path=checkpoint,
            checkpoint_sha256=(
                None
                if fresh
                else job["resume_input"]["checkpoint_sha256"]
            ),
        )
        # This is intentionally the final pre-execution reconstruction.
        effective_contract = build_effective_execution_contract(
            job=job,
            derived_protocol_payload=protocol.to_dict(),
        )
        if (
            effective_contract
            != job["effective_execution_contract"]
            or effective_contract["sha256"]
            != job["effective_execution_contract_sha256"]
            or effective_contract["scientific_settings_sha256"]
            != job["scientific_settings_sha256"]
            or effective_contract["operational_settings_sha256"]
            != job["operational_settings_sha256"]
        ):
            raise PackageContractError(
                "Immediate pre-run effective contract reconstruction "
                "drifted."
            )
        from pipelines.static_adapt.ra_adapt import run_ra_adapt

        original = Path.cwd()
        os.chdir(
            (
                source_root
                / safe_relative_path(
                    job["source_protocol"]["path"],
                    label="source protocol path",
                )
            ).parent.parent
        )
        try:
            result = run_ra_adapt(
                problem,
                protocol,
                operational_controls=controls,
            )
        finally:
            os.chdir(original)
        execution_manifest = _write_result_artifacts(
            staging=staging,
            job=job,
            authorization=authorization,
            clearance=clearance,
            protocol=protocol,
            result=result,
            effective_contract=effective_contract,
        )
        os.rename(staging, output_dir)
    receipt = digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_r70_"
                "continuation_worker_receipt_v2"
            ),
            "status": "passed",
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "collision_clearance_sha256": (
                clearance["sha256"]
                if clearance is not None
                else None
            ),
            "effective_execution_contract_sha256": (
                job["effective_execution_contract_sha256"]
            ),
            "scientific_settings_sha256": job[
                "scientific_settings_sha256"
            ],
            "operational_settings_sha256": job[
                "operational_settings_sha256"
            ],
            "execution_manifest_sha256": (
                execution_manifest["sha256"]
            ),
            "output_dir": output_dir.as_posix(),
            "output_files": [
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(
                    output_dir.iterdir(), key=lambda item: item.name
                )
                if path.is_file()
            ],
        }
    )
    base_runtime._write_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument(
        "--execution-authorization",
        "--authorization",
        dest="authorization",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--collision-clearance",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = run_cell(
            job_path=args.job.resolve(),
            authorization_path=args.authorization.resolve(),
            collision_clearance_path=(
                args.collision_clearance.resolve()
                if args.collision_clearance is not None
                else None
            ),
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
