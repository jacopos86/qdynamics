#!/usr/bin/env python3
"""Validate the sealed cumulative-relative strong--strong r70 package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ALLOWED_OPERATIONAL_SOURCE_DELTAS,
    CAMPAIGN_ID,
    CANDIDATE_REPRESENTATION,
    DERIVED_PROTOCOL_CHANGED_PATHS,
    EXECUTION_ID,
    PACKAGE_ID,
    PLATEAU_COMPARISON,
    PLATEAU_RATIO_THRESHOLD,
    PLATEAU_TRIGGER,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_CONTRACT_SHA256,
    SOURCE_HORIZON,
    SOURCE_PROTOCOL_CANONICAL_SHA256,
    TARGET_HORIZON,
    PackageContractError,
    load_json,
    safe_relative_path,
    scalar_differences,
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


def _verify_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    relative = safe_relative_path(binding.get("path"), label=f"{label} path")
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped package.") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    observed = verify_self_digest(payload, label=label)
    if observed != binding.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _validate_tar(
    *,
    archive_path: Path,
    rows: list[Any],
    label: str,
) -> dict[str, Any]:
    expected: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, label=f"{label} member")
        name = safe_relative_path(
            row.get("path"), label=f"{label} member path"
        ).as_posix()
        if name in expected:
            raise PackageContractError(f"Duplicate {label} binding: {name}")
        expected[name] = row
    observed: set[str] = set()
    decompressed_bytes = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            row = expected.get(member.name)
            if (
                row is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"Unexpected {label} member: {member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable {label} member: {member.name}"
                )
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            if (
                size != member.size
                or digest.hexdigest() != row.get("sha256")
            ):
                raise PackageContractError(
                    f"{label} member digest drifted: {member.name}"
                )
            decompressed_bytes += size
            observed.add(member.name)
    if observed != set(expected):
        raise PackageContractError(f"{label} member closure drifted.")
    return {
        "member_count": len(observed),
        "decompressed_bytes": decompressed_bytes,
    }


def validate(*, deep_resume: bool) -> dict[str, Any]:
    if any(
        path.name == "__pycache__" or path.suffix == ".pyc"
        for path in PACKAGE_DIR.rglob("*")
    ):
        raise PackageContractError(
            "Unbound Python bytecode is forbidden in the transferred package."
        )
    manifest = load_json(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema")
        != "paper_i_ra_adapt_cumulative_relative_r70_package_manifest_v1"
        or manifest.get("status") != "passed_inert_one_row"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("execution_id") != EXECUTION_ID
        or manifest.get("row_count") != 1
        or manifest.get("source_horizon") != SOURCE_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Package manifest contract drifted.")

    for row in _sequence(
        manifest.get("control_files"), label="control files"
    ):
        _verify_binding(
            _mapping(row, label="control file"), label="control file"
        )
    documents: dict[str, dict[str, Any]] = {}
    for key in (
        "derived_protocol",
        "source_archive_manifest",
        "source_delta_receipt",
        "resume_input",
        "source_lock_audit",
        "execution_plan",
        "job",
    ):
        _path, payload = _verify_binding(
            _mapping(manifest.get(key), label=f"{key} binding"),
            label=key,
            canonical=True,
        )
        assert payload is not None
        documents[key] = payload
    source_archive, _ = _verify_binding(
        _mapping(manifest.get("source_archive"), label="source archive"),
        label="source archive",
    )
    for role, raw in _mapping(
        manifest.get("lineage"), label="lineage"
    ).items():
        _verify_binding(
            _mapping(raw, label=f"lineage {role}"),
            label=f"lineage {role}",
            canonical=role not in {"r50_result", "r50_summary"},
        )

    job = documents["job"]
    plan = documents["execution_plan"]
    audit = documents["source_lock_audit"]
    delta = documents["source_delta_receipt"]
    resume = documents["resume_input"]
    source_manifest = documents["source_archive_manifest"]
    # Resolve the repository root without trusting cwd.
    repo_root = PACKAGE_DIR
    while not (repo_root / "AGENTS.md").is_file():
        if repo_root.parent == repo_root:
            raise PackageContractError("Repository root not found.")
        repo_root = repo_root.parent
    source_protocol_path = (
        repo_root
        / "output/local_runs/"
        "paper_i_ra_adapt_cumulative_plateau_pair_r20_local_20260731_v1/"
        "materialization/protocols/"
        "core__strong_strong_u8__nph7__ra_singleton_plateau.json"
    )
    source_payload = load_json(source_protocol_path, label="source protocol")
    verify_self_digest(source_payload, label="source protocol")
    derived = documents["derived_protocol"]
    changed = sorted(
        ".".join(str(component) for component in path)
        for path, _before, _after in scalar_differences(
            source_payload, derived
        )
    )
    invariants = derived["route_contract"]["semantic_invariants"]
    if (
        source_payload.get("sha256") != SOURCE_PROTOCOL_CANONICAL_SHA256
        or changed != list(DERIVED_PROTOCOL_CHANGED_PATHS)
        or derived.get("horizon") != TARGET_HORIZON
        or derived.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or derived.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or derived.get("candidate_representation")
        != CANDIDATE_REPRESENTATION
        or derived["route_contract"]["sha256"] != ROUTE_CONTRACT_SHA256
        or invariants["plateau_cumulative_decrease_ratio_threshold"]
        != PLATEAU_RATIO_THRESHOLD
        or invariants["plateau_threshold_comparison"]
        != PLATEAU_COMPARISON
        or invariants["plateau_trigger_source"] != PLATEAU_TRIGGER
        or plan.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or audit.get("changed_protocol_paths")
        != list(DERIVED_PROTOCOL_CHANGED_PATHS)
        or audit.get("non_swept_settings_diff") != []
        or job.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
    ):
        raise PackageContractError("Horizon-only scientific delta drifted.")

    changed_sources = _sequence(
        delta.get("changed_members"), label="source delta members"
    )
    if (
        delta.get("status") != "passed_operational_only"
        or {row.get("path") for row in changed_sources}
        != set(ALLOWED_OPERATIONAL_SOURCE_DELTAS)
        or delta.get("scientific_settings_changed") != []
        or delta.get("route_contract_changed") is not False
        or audit.get("scientific_settings_changed_by_source_delta") != []
    ):
        raise PackageContractError("Source-delta attestation drifted.")

    source_rows = _sequence(
        source_manifest.get("members"), label="source archive members"
    )
    if (
        source_manifest.get("archive") != manifest.get("source_archive")
        or len(source_rows) != source_manifest.get("member_count")
    ):
        raise PackageContractError("Source archive manifest drifted.")
    source_scan = _validate_tar(
        archive_path=source_archive,
        rows=source_rows,
        label="source archive",
    )
    resume_archive, _ = _verify_binding(
        _mapping(resume.get("archive"), label="resume archive"),
        label="resume archive",
    )
    resume_rows = _sequence(resume.get("members"), label="resume members")
    if (
        len(resume_rows) != 3
        or resume.get("member_count") != 3
        or resume.get("pointer_closed") is not True
        or resume.get("checkpoint_sha256")
        != "b8186aabb56c8fee9ff71d5a6a9c6f5a7c18ea42e36431b65d54fca245386811"
    ):
        raise PackageContractError("Resume manifest drifted.")
    resume_scan = _validate_tar(
        archive_path=resume_archive,
        rows=resume_rows,
        label="resume archive",
    )

    preflight: dict[str, Any] | None = None
    if deep_resume:
        command = [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "run_cell.py"),
            "--job",
            str(PACKAGE_DIR / "job.json"),
            "--preflight",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise PackageContractError(
                "Deep source-locked resume preflight failed: "
                + completed.stderr.strip()
            )
        value = json.loads(completed.stdout)
        if not isinstance(value, dict) or value.get("status") != "passed":
            raise PackageContractError("Deep preflight returned bad receipt.")
        preflight = value
    return {
        "status": "passed",
        "package_manifest_sha256": manifest["sha256"],
        "job_spec_sha256": job["sha256"],
        "derived_protocol_sha256": derived["sha256"],
        "source_archive": source_scan,
        "resume_archive": resume_scan,
        "deep_resume_preflight": preflight,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-resume", action="store_true")
    args = parser.parse_args()
    try:
        payload = validate(deep_resume=args.deep_resume)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
