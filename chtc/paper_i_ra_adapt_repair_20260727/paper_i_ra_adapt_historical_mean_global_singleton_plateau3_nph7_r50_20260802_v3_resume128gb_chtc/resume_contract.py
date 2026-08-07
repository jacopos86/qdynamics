#!/usr/bin/env python3
"""Closed contract for the three memory-repair accepted-state resumes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_resume128gb_v3"
)
RUN_CLASS = "diagnostic"
TARGET_HORIZON = 50
SOURCE_CLUSTER_ID = 9400751

PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_chtc"
)
INPUTS_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_inputs_v1"
)
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v2_chtc"
)
SNAPSHOT_SOURCE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "live_safety_snapshots_20260802"
)

SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "1b5bf20d8754fdf66a48727d857a7ef2e090e5f541afa303e453bbb4ea3ec8c3"
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "fe7fd6f5f572c3ca90dbf43ec43c69f35282d4c699cd271d8cd6564555bb495f"
)
SOURCE_ARCHIVE_SHA256 = (
    "7e7fa374f629ce684035d318176f354b24cfdf7cf4ac9548be921c790bf57d01"
)
SOURCE_RUNNER_SHA256 = (
    "8694d5b241168fbad387c64b92648da1c992ce74e0a30e0dc1703f7cf3ed073e"
)
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1"
)
ROUTE_CONTRACT_SHA256 = (
    "69af64db5bbaf5b811685b8353b82b748dc13d16306e4c08ddfe5ffde07f301b"
)

RESOURCE_ENVELOPE = {
    "request_cpus": 4,
    "request_memory_mb": 131_072,
    "request_disk_mb": 81_920,
    "max_runtime_seconds": 259_200,
    "basis": (
        "memory_limit_repair_plus_full_json_checkpoint_and_ledger_"
        "hydration_headroom_v1"
    ),
}

ROW_SPECS = (
    {
        "proc_id": 0,
        "regime_id": "weak_strong",
        "source_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__weak_strong__"
            "nph7__ra_global_singleton_plateau"
        ),
        "snapshot_name": (
            "9400751.0__weak_strong__20260802T192821Z.tar.gz"
        ),
        "validation_name": (
            "9400751.0__weak_strong__20260802T192821Z.validation.json"
        ),
        "validation_file_sha256": (
            "3ef3433d518aa4ce3b24bf6d83379df0d9129174d01a0e3e10ade50d72e95a02"
        ),
        "archive_sha256": (
            "06efd350c3c188d8e5af75f385ede22d340b71355c48b38516294ec17af6b6c7"
        ),
        "archive_size_bytes": 1_227_663_445,
        "resume_controller_round": 35,
    },
    {
        "proc_id": 1,
        "regime_id": "intermediate_strong",
        "source_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__"
            "intermediate_strong__nph7__ra_global_singleton_plateau"
        ),
        "snapshot_name": (
            "9400751.1__intermediate_strong__20260802T192821Z.tar.gz"
        ),
        "validation_name": (
            "9400751.1__intermediate_strong__20260802T192821Z.validation.json"
        ),
        "validation_file_sha256": (
            "d27f1ef5a5b3d7d0075819011bf979ae2a6c696339571a4c8b5bb2597d146d8a"
        ),
        "archive_sha256": (
            "2daba25929aae14889e22b968e8da24c0e2c396749b155bd985d9363e480c049"
        ),
        "archive_size_bytes": 1_083_965_298,
        "resume_controller_round": 31,
    },
    {
        "proc_id": 2,
        "regime_id": "strong_strong_u8",
        "source_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__"
            "strong_strong_u8__nph7__ra_global_singleton_plateau"
        ),
        "snapshot_name": (
            "9400751.2__strong_strong__20260802T192821Z.tar.gz"
        ),
        "validation_name": (
            "9400751.2__strong_strong__20260802T192821Z.validation.json"
        ),
        "validation_file_sha256": (
            "b05e52c3cbe07aff722efe9d7f1b91c4365d3b9a9232c39db9807fd4b17541d5"
        ),
        "archive_sha256": (
            "80941a839ae950b90e30c1da7b82aefcee8ce61d87790329fbb8117454145ed9"
        ),
        "archive_size_bytes": 1_142_566_246,
        "resume_controller_round": 17,
    },
)

CONTROL_FILES = (
    "resume_contract.py",
    "build_package.py",
    "run_resume_cell.py",
    "validate_package.py",
)
GENERATED_PATHS = (
    "jobs",
    "resume_inputs_manifest.json",
    "execution_plan.json",
    "queue.tsv",
    "package_manifest.json",
)
PACKAGE_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "package_manifest_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "execution_authorization_v1"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ResumeContractError(ValueError):
    """Raised when the sealed operational resume contract drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResumeContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ResumeContractError(f"{label} must be a JSON object.")
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = canonical_sha256(value)
    if value.get("sha256") != observed:
        raise ResumeContractError(f"{label} canonical digest drifted.")
    return observed


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ResumeContractError(f"{label} must be a relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise ResumeContractError(f"Unsafe {label}: {value}")
    return Path(*pure.parts)


def repo_root_from_script(path: str | Path) -> Path:
    for candidate in Path(path).resolve().parents:
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines"
        ).is_dir():
            return candidate
    raise ResumeContractError("Active repository root was not found.")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ResumeContractError(f"Unsafe bound file: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=path.name)
    canonical = verify_self_digest(payload, label=path.name)
    return {
        **file_binding(path, relative_to=relative_to),
        "canonical_sha256": canonical,
    }


def execution_id(source_execution_id: str, depth: int) -> str:
    return f"{source_execution_id}__resume_from_d{depth}_to_r50_v1"


def _verify_exact_file(
    path: Path, *, size_bytes: int, sha256: str, label: str, hash_file: bool
) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != size_bytes
        or (hash_file and sha256_file(path) != sha256)
    ):
        raise ResumeContractError(f"{label} exact bytes drifted: {path}")


def validate_source_package(repo_root: Path) -> dict[str, Any]:
    root = repo_root / SOURCE_PACKAGE_RELATIVE
    manifest_path = root / "package_manifest.json"
    _verify_exact_file(
        manifest_path,
        size_bytes=manifest_path.stat().st_size,
        sha256=SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
        label="source package manifest",
        hash_file=True,
    )
    manifest = load_json(manifest_path, label="source package manifest")
    if (
        verify_self_digest(manifest, label="source package manifest")
        != SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("package_id")
        != (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_20260802_v2_chtc"
        )
        or manifest.get("row_count") != 3
    ):
        raise ResumeContractError("Source package identity drifted.")
    source_archive = root / "source/source_locked.tar.gz"
    if sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise ResumeContractError("Source archive bytes drifted.")
    if sha256_file(root / "run_cell.py") != SOURCE_RUNNER_SHA256:
        raise ResumeContractError("Source runner bytes drifted.")
    return {"root": root, "manifest": manifest}


def _source_row(
    source: Mapping[str, Any], *, source_execution_id: str
) -> dict[str, Any]:
    rows = source["manifest"].get("jobs")
    if not isinstance(rows, list):
        raise ResumeContractError("Source job bindings are absent.")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("execution_id") == source_execution_id
    ]
    if len(matches) != 1:
        raise ResumeContractError("Source job binding is not unique.")
    return dict(matches[0])


def _snapshot_binding(
    repo_root: Path, spec: Mapping[str, Any], *, hash_archive: bool
) -> dict[str, Any]:
    source_validation = (
        repo_root / SNAPSHOT_SOURCE_RELATIVE / str(spec["validation_name"])
    )
    _verify_exact_file(
        source_validation,
        size_bytes=source_validation.stat().st_size,
        sha256=str(spec["validation_file_sha256"]),
        label="snapshot validation",
        hash_file=True,
    )
    validation = load_json(source_validation, label="snapshot validation")
    staged_archive = repo_root / INPUTS_RELATIVE / str(spec["snapshot_name"])
    source_archive = (
        repo_root / SNAPSHOT_SOURCE_RELATIVE / str(spec["snapshot_name"])
    )
    for label, path in (
        ("source snapshot archive", source_archive),
        ("staged snapshot archive", staged_archive),
    ):
        _verify_exact_file(
            path,
            size_bytes=int(spec["archive_size_bytes"]),
            sha256=str(spec["archive_sha256"]),
            label=label,
            hash_file=hash_archive,
        )
    if (
        validation.get("schema")
        != "paper_i_live_checkpoint_snapshot_validation_v1"
        or validation.get("validation") != "passed"
        or validation.get("archive_sha256") != spec["archive_sha256"]
        or validation.get("archive_size_bytes") != spec["archive_size_bytes"]
        or validation.get("checkpoint_depth")
        != spec["resume_controller_round"]
    ):
        raise ResumeContractError("Snapshot validation identity drifted.")
    members = validation.get("members")
    pointers = validation.get("pointers")
    if not isinstance(members, Mapping) or not isinstance(pointers, Mapping):
        raise ResumeContractError("Snapshot pointer closure is absent.")
    ledger = pointers.get("ledger")
    resume = pointers.get("resume")
    if not isinstance(ledger, Mapping) or not isinstance(resume, Mapping):
        raise ResumeContractError("Snapshot sidecar pointers are malformed.")
    roles = {
        "checkpoint.json": "checkpoint",
        str(ledger.get("path")): "estimator_ledger_checkpoint",
        str(resume.get("path")): "verified_resume_sidecar",
    }
    if set(roles) != set(members) or len(roles) != 3:
        raise ResumeContractError("Snapshot is not a pointer-closed triplet.")
    member_rows: list[dict[str, Any]] = []
    for name, role in roles.items():
        row = members[name]
        if not isinstance(row, Mapping) or not _HEX64.fullmatch(
            str(row.get("sha256", ""))
        ):
            raise ResumeContractError("Snapshot member binding is malformed.")
        if role == "estimator_ledger_checkpoint" and row.get(
            "sha256"
        ) != ledger.get("sha256"):
            raise ResumeContractError("Ledger pointer digest drifted.")
        if role == "verified_resume_sidecar" and row.get(
            "sha256"
        ) != resume.get("sha256"):
            raise ResumeContractError("Resume pointer digest drifted.")
        member_rows.append(
            {
                "role": role,
                "path": name,
                "sha256": str(row["sha256"]),
                "size_bytes": int(row["size_bytes"]),
            }
        )
    member_rows.sort(key=lambda row: str(row["role"]))
    checkpoint = next(
        row for row in member_rows if row["role"] == "checkpoint"
    )
    return {
        "archive": {
            "path": staged_archive.relative_to(repo_root).as_posix(),
            "sha256": str(spec["archive_sha256"]),
            "size_bytes": int(spec["archive_size_bytes"]),
        },
        "source_archive": {
            "path": source_archive.relative_to(repo_root).as_posix(),
            "sha256": str(spec["archive_sha256"]),
            "size_bytes": int(spec["archive_size_bytes"]),
        },
        "validation": {
            "path": source_validation.relative_to(repo_root).as_posix(),
            "sha256": str(spec["validation_file_sha256"]),
            "size_bytes": source_validation.stat().st_size,
        },
        "checkpoint_path": "checkpoint.json",
        "checkpoint_sha256": checkpoint["sha256"],
        "member_count": 3,
        "members": member_rows,
        "pointer_closed": True,
        "validation_status": "passed",
        "resume_controller_round": int(spec["resume_controller_round"]),
    }


def expected_jobs(
    repo_root: Path, *, hash_archives: bool
) -> list[dict[str, Any]]:
    source = validate_source_package(repo_root)
    package_root = repo_root / PACKAGE_RELATIVE
    jobs: list[dict[str, Any]] = []
    for spec in ROW_SPECS:
        source_execution_id = str(spec["source_execution_id"])
        source_row = _source_row(source, source_execution_id=source_execution_id)
        source_job_path = source["root"] / safe_relative_path(
            source_row.get("path"), label="source job path"
        )
        source_job = load_json(source_job_path, label="source job")
        if (
            verify_self_digest(source_job, label="source job")
            != source_row.get("canonical_sha256")
            or sha256_file(source_job_path) != source_row.get("sha256")
            or source_job.get("execution_id") != source_execution_id
            or source_job.get("regime_id") != spec["regime_id"]
            or source_job.get("nph") != 7
            or source_job.get("route_contract_sha256")
            != ROUTE_CONTRACT_SHA256
            or source_job.get("fresh_start_contract", {}).get("kind")
            != "fresh_start"
        ):
            raise ResumeContractError("Source job scientific identity drifted.")
        protocol_path = source["root"] / safe_relative_path(
            source_job.get("protocol_path"), label="source protocol path"
        )
        protocol = load_json(protocol_path, label="source protocol")
        if (
            verify_self_digest(protocol, label="source protocol")
            != source_job.get("protocol_sha256")
            or sha256_file(protocol_path)
            != source_job.get("protocol_file_sha256")
            or protocol.get("horizon") != TARGET_HORIZON
            or protocol.get("request", {})
            .get("execution", {})
            .get("stop", {})
            .get("maximum_controller_rounds")
            != TARGET_HORIZON
            or protocol.get("request", {})
            .get("execution", {})
            .get("resume", {})
            .get("kind")
            != "fresh_start"
            or protocol.get("route_contract", {}).get("route_profile")
            != ROUTE_PROFILE
            or protocol.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
        ):
            raise ResumeContractError("Source protocol scientific identity drifted.")
        resume_input = _snapshot_binding(
            repo_root, spec, hash_archive=hash_archives
        )
        depth = int(spec["resume_controller_round"])
        identifier = execution_id(source_execution_id, depth)
        jobs.append(
            digested(
                {
                    "schema": JOB_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "campaign_id": CAMPAIGN_ID,
                    "execution_id": identifier,
                    "source_execution_id": source_execution_id,
                    "execution_mode": "authenticated_accepted_state_resume_to_50",
                    "run_class": RUN_CLASS,
                    "execution_target": "chtc",
                    "regime_id": spec["regime_id"],
                    "nph": 7,
                    "route_id": "ra_global_singleton_plateau_commutation",
                    "route_profile": ROUTE_PROFILE,
                    "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                    "source_cluster_id": SOURCE_CLUSTER_ID,
                    "source_proc_id": int(spec["proc_id"]),
                    "source_job_preserved_held": True,
                    "target_horizon": TARGET_HORIZON,
                    "source_package": {
                        "path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                        "manifest_sha256": SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
                        "manifest_file_sha256": SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
                    },
                    "source_job": {
                        "path": source_job_path.relative_to(repo_root).as_posix(),
                        "sha256": str(source_row["sha256"]),
                        "canonical_sha256": str(source_row["canonical_sha256"]),
                        "size_bytes": source_job_path.stat().st_size,
                    },
                    "source_protocol": {
                        "path": protocol_path.relative_to(repo_root).as_posix(),
                        "sha256": sha256_file(protocol_path),
                        "canonical_sha256": str(protocol["sha256"]),
                        "size_bytes": protocol_path.stat().st_size,
                    },
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "source_runner_sha256": SOURCE_RUNNER_SHA256,
                    "resume_input": resume_input,
                    "resources": dict(RESOURCE_ENVELOPE),
                    "scientific_protocol_sha256": str(protocol["sha256"]),
                    "scientific_protocol_changed": False,
                    "scientific_settings_changed": [],
                    "operational_changes": [
                        "fresh_start_to_accepted_state_resume",
                        "request_memory_to_131072_mb",
                        "request_disk_to_81920_mb",
                        "noncolliding_output_paths",
                    ],
                    "expected_output_root": f"runs/{identifier}",
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                }
            )
        )
    return jobs


def package_root(repo_root: Path) -> Path:
    root = repo_root / PACKAGE_RELATIVE
    if not root.is_dir() or root.is_symlink():
        raise ResumeContractError("Resume package root is unavailable.")
    return root
