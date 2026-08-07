#!/usr/bin/env python3
"""Closed contract for exact-prefix global-singleton r70 resumes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r70_20260804_v1_resume256gb_loaderfix_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r70_exact_prefix_49_45_31_resume256gb_loaderfix_v2"
)
PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / PACKAGE_ID
RUN_CLASS = "diagnostic"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70
SOURCE_CLUSTER_ID = 9401106

SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v2_chtc"
)
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "1b5bf20d8754fdf66a48727d857a7ef2e090e5f541afa303e453bbb4ea3ec8c3"
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "fe7fd6f5f572c3ca90dbf43ec43c69f35282d4c699cd271d8cd6564555bb495f"
)
SOURCE_RUNNER_SHA256 = (
    "8694d5b241168fbad387c64b92648da1c992ce74e0a30e0dc1703f7cf3ed073e"
)
SOURCE_ARCHIVE_SHA256 = (
    "7e7fa374f629ce684035d318176f354b24cfdf7cf4ac9548be921c790bf57d01"
)

LOADER_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_loaderfix_v2_chtc"
)
LOADER_PACKAGE_MANIFEST_FILE_SHA256 = (
    "0b81923cbc691fb18ca58bb78da73d6ce9ba501e6717ee315b2a2fb8744b293d"
)
LOADER_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "84d8f7bdcc79e986c8bbd22af8f3c1c5ed2d5c1b95aeb1e84affb5c3ae87e1a1"
)
LOADER_RUNNER_SHA256 = (
    "ebaef5b523aa6d425d112c290c6654851db54cd6dbabfe32123fc3975ed42023"
)
RESUME_LOADER_PATCH_PATH = "pipelines/static_adapt/sr_snake/_resume.py"
RESUME_LOADER_BEFORE_SHA256 = (
    "6d3753f22071cae21eb5eb006e634655be0fb4a9ec60054d61dfef2a3625e37f"
)
RESUME_LOADER_AFTER_SHA256 = (
    "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
)
IMPLEMENTATION_REPAIR_ID = (
    "accepted_round_current_checkpoint_receipt_loader_fix_v2"
)

SNAPSHOT_ROOT_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "live_safety_snapshots_20260803_9401106"
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
    "request_memory_mb": 262_144,
    "request_disk_mb": 102_400,
    "max_runtime_seconds": 259_200,
    "basis": (
        "observed_128gib_cgroup_memory_failure_plus_exact_prefix_"
        "hydration_and_round70_growth_headroom_v1"
    ),
}

HORIZON_CHANGED_PATHS = (
    "horizon",
    "request.execution.stop.maximum_controller_rounds",
    "sha256",
    "stopping_rule.maximum_controller_rounds",
)
SCIENTIFIC_SETTINGS_CHANGED = ("maximum_controller_rounds_50_to_70",)

ROW_SPECS = (
    {
        "proc_id": 0,
        "regime_id": "weak_strong",
        "resume_controller_round": 49,
        "source_resume_controller_round": 35,
        "base_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__weak_strong__"
            "nph7__ra_global_singleton_plateau"
        ),
        "predecessor_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__weak_strong__"
            "nph7__ra_global_singleton_plateau__resume_from_d35_to_r50_"
            "loaderfix_v2"
        ),
        "stem": "9401106.0__weak_strong__20260804T012503Z",
        "archive_sha256": (
            "c0589600744902f276c479fa05d7de53b55345b11221bd544de5183d8eabaf9c"
        ),
        "archive_size_bytes": 4_903_485_221,
        "validation_file_sha256": (
            "259270606fff3b4893dbf4eec1dffdbebd4faa2d0296e45881002cff8f5c2a09"
        ),
        "projection_file_sha256": (
            "a9416249eb8d9d23edb43fcfadad03005a016fb4c9100f0b386c60f377f93ad1"
        ),
    },
    {
        "proc_id": 1,
        "regime_id": "intermediate_strong",
        "resume_controller_round": 45,
        "source_resume_controller_round": 31,
        "base_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__"
            "intermediate_strong__nph7__ra_global_singleton_plateau"
        ),
        "predecessor_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__"
            "intermediate_strong__nph7__ra_global_singleton_plateau__"
            "resume_from_d31_to_r50_loaderfix_v2"
        ),
        "stem": "9401106.1__intermediate_strong__20260803T222159Z",
        "archive_sha256": (
            "3a681f5600f32bd5f6a8196afeae8ebba588268d9a5a29aac22ce79624334848"
        ),
        "archive_size_bytes": 3_776_852_651,
        "validation_file_sha256": (
            "be7d37fef6923bf0f9d4fad382a1a661a5e0767777fbb88e7ad29e7354c94629"
        ),
        "projection_file_sha256": (
            "448587ab3d56ffd0fa5d57c62e6331bf4a3d2688135370dadae0931db0103082"
        ),
    },
    {
        "proc_id": 2,
        "regime_id": "strong_strong_u8",
        "resume_controller_round": 31,
        "source_resume_controller_round": 17,
        "base_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__"
            "strong_strong_u8__nph7__ra_global_singleton_plateau"
        ),
        "predecessor_execution_id": (
            "historical_mean_global_singleton_v2_nph7_r50__"
            "strong_strong_u8__nph7__ra_global_singleton_plateau__"
            "resume_from_d17_to_r50_loaderfix_v2"
        ),
        "stem": "9401106.2__strong_strong_u8__20260803T222743Z",
        "archive_sha256": (
            "ec80ba62988305a74d37d8dd780410bbcbab788f3423a516046fa11c1be5f393"
        ),
        "archive_size_bytes": 3_905_062_171,
        "validation_file_sha256": (
            "8ab1c507a232f86163fc5c3f31d7e662e0d60e38d1e86d507113e83591c01ddc"
        ),
        "projection_file_sha256": (
            "5183701e05fd45dbf6180e060743ce5bae6b846057437a20fd4e1e45f5cb5940"
        ),
    },
)

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_resume_cell.py",
    "validate_package.py",
)
GENERATED_PATHS = (
    "protocols",
    "jobs",
    "resume_inputs_manifest.json",
    "source_lock_audit.json",
    "execution_plan.json",
    "queue.tsv",
    "package_manifest.json",
)
PACKAGE_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "resume256gb_package_manifest_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "resume256gb_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "resume256gb_execution_authorization_v1"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PackageContractError(ValueError):
    """Raised when the exact-prefix continuation contract drifts."""


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
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = canonical_sha256(value)
    if value.get("sha256") != observed:
        raise PackageContractError(f"{label} canonical digest drifted.")
    return observed


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise PackageContractError(f"Unsafe {label}: {value}")
    return Path(*pure.parts)


def repo_root_from_script(path: str | Path) -> Path:
    for candidate in Path(path).resolve().parents:
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines"
        ).is_dir():
            return candidate
    raise PackageContractError("Active repository root was not found.")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Unsafe bound file: {path}")
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


def execution_id(spec: Mapping[str, Any]) -> str:
    return (
        "historical_mean_global_singleton_v2_nph7_r70__"
        f"{spec['regime_id']}__nph7__ra_global_singleton_plateau__"
        f"resume_from_d{spec['resume_controller_round']}_to_r70_"
        "256gb_loaderfix_v2"
    )


def scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        rows: list[tuple[tuple[str | int, ...], Any, Any]] = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                rows.append(((*path, str(key)), before.get(key), after.get(key)))
            else:
                rows.extend(
                    scalar_differences(
                        before[key], after[key], path=(*path, str(key))
                    )
                )
        return rows
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [(path, before, after)]
        rows = []
        for index, (left, right) in enumerate(zip(before, after)):
            rows.extend(
                scalar_differences(left, right, path=(*path, index))
            )
        return rows
    return [] if before == after else [(path, before, after)]


def implementation_repair() -> dict[str, Any]:
    return {
        "repair_id": IMPLEMENTATION_REPAIR_ID,
        "path": RESUME_LOADER_PATCH_PATH,
        "before_sha256": RESUME_LOADER_BEFORE_SHA256,
        "after_sha256": RESUME_LOADER_AFTER_SHA256,
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
    }


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _verified_manifest(
    repo_root: Path,
    relative: Path,
    *,
    file_sha256: str,
    canonical_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    path = repo_root / relative / "package_manifest.json"
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != file_sha256
    ):
        raise PackageContractError(f"{label} manifest bytes drifted.")
    payload = load_json(path, label=f"{label} manifest")
    if verify_self_digest(payload, label=f"{label} manifest") != canonical_sha256:
        raise PackageContractError(f"{label} manifest identity drifted.")
    return path, payload


def _bound_job(
    *,
    repo_root: Path,
    package_relative: Path,
    manifest: Mapping[str, Any],
    execution_id_value: str,
    label: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    bindings = manifest.get("jobs")
    if not isinstance(bindings, list):
        raise PackageContractError(f"{label} job bindings are absent.")
    matches = [
        row
        for row in bindings
        if isinstance(row, Mapping)
        and row.get("execution_id") == execution_id_value
    ]
    if len(matches) != 1:
        raise PackageContractError(f"{label} job binding is not unique.")
    binding = dict(matches[0])
    path = repo_root / package_relative / safe_relative_path(
        binding.get("path"), label=f"{label} job path"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} job bytes drifted.")
    payload = load_json(path, label=f"{label} job")
    if verify_self_digest(payload, label=f"{label} job") != binding.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} job digest drifted.")
    return path, payload, binding


def _source_protocol(
    repo_root: Path,
    source_job: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = repo_root / SOURCE_PACKAGE_RELATIVE / safe_relative_path(
        source_job.get("protocol_path"), label="source protocol path"
    )
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != source_job.get("protocol_file_sha256")
    ):
        raise PackageContractError("Source protocol bytes drifted.")
    payload = load_json(path, label="source protocol")
    if (
        verify_self_digest(payload, label="source protocol")
        != source_job.get("protocol_sha256")
        or payload.get("horizon") != SOURCE_HORIZON
        or payload.get("request", {})
        .get("execution", {})
        .get("stop", {})
        .get("maximum_controller_rounds")
        != SOURCE_HORIZON
        or payload.get("stopping_rule", {}).get("maximum_controller_rounds")
        != SOURCE_HORIZON
        or payload.get("route_contract", {}).get("route_profile")
        != ROUTE_PROFILE
        or payload.get("route_contract", {}).get("sha256")
        != ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError("Source protocol scientific identity drifted.")
    return path, payload


def derived_protocol(source: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(source)))
    payload.pop("sha256", None)
    payload["horizon"] = TARGET_HORIZON
    payload["request"]["execution"]["stop"][
        "maximum_controller_rounds"
    ] = TARGET_HORIZON
    payload["stopping_rule"][
        "maximum_controller_rounds"
    ] = TARGET_HORIZON
    payload = digested(payload)
    changed = tuple(
        sorted(
            ".".join(str(component) for component in path)
            for path, _before, _after in scalar_differences(source, payload)
        )
    )
    if changed != tuple(sorted(HORIZON_CHANGED_PATHS)):
        raise PackageContractError(
            f"Source-to-r70 protocol delta drifted: {changed}"
        )
    return payload


def _snapshot_binding(
    repo_root: Path,
    spec: Mapping[str, Any],
    *,
    hash_archive: bool,
) -> dict[str, Any]:
    stem = str(spec["stem"])
    archive = repo_root / SNAPSHOT_ROOT_RELATIVE / f"{stem}.tar.gz"
    validation_path = repo_root / SNAPSHOT_ROOT_RELATIVE / f"{stem}.validation.json"
    projection_path = (
        repo_root
        / SNAPSHOT_ROOT_RELATIVE
        / f"{stem}.authenticated_v2.live_projection.json"
    )
    if (
        not archive.is_file()
        or archive.is_symlink()
        or archive.stat().st_size != int(spec["archive_size_bytes"])
        or (hash_archive and sha256_file(archive) != spec["archive_sha256"])
    ):
        raise PackageContractError(f"Snapshot archive drifted: {stem}")
    if (
        not validation_path.is_file()
        or validation_path.is_symlink()
        or sha256_file(validation_path) != spec["validation_file_sha256"]
    ):
        raise PackageContractError(f"Snapshot validation drifted: {stem}")
    validation = load_json(validation_path, label="snapshot validation")
    if (
        validation.get("schema")
        != "paper_i_live_checkpoint_snapshot_validation_v1"
        or validation.get("validation") != "passed"
        or validation.get("archive_sha256") != spec["archive_sha256"]
        or validation.get("archive_size_bytes") != spec["archive_size_bytes"]
        or validation.get("checkpoint_depth")
        != spec["resume_controller_round"]
    ):
        raise PackageContractError(f"Snapshot validation identity drifted: {stem}")
    if (
        not projection_path.is_file()
        or projection_path.is_symlink()
        or sha256_file(projection_path) != spec["projection_file_sha256"]
    ):
        raise PackageContractError(f"Snapshot projection drifted: {stem}")
    projection = load_json(projection_path, label="authenticated projection")
    verify_self_digest(projection, label="authenticated projection")
    binding = projection.get("snapshot_execution_binding")
    checks = projection.get("snapshot_validation")
    if (
        projection.get("status") != "passed_authenticated_live_partial"
        or projection.get("cluster_id") != SOURCE_CLUSTER_ID
        or projection.get("proc_id") != spec["proc_id"]
        or projection.get("regime_id") != spec["regime_id"]
        or projection.get("live_controller_round")
        != spec["resume_controller_round"]
        or projection.get("route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or not isinstance(binding, Mapping)
        or binding.get("execution_id") != spec["predecessor_execution_id"]
        or binding.get("route_and_problem_binding_passed") is not True
        or binding.get("strict_numerical_replay_passed") is not True
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise PackageContractError(
            f"Authenticated projection closure drifted: {stem}"
        )
    raw_members = validation.get("members")
    pointers = validation.get("pointers")
    if not isinstance(raw_members, Mapping) or not isinstance(pointers, Mapping):
        raise PackageContractError("Snapshot member closure is absent.")
    ledger = pointers.get("ledger")
    resume = pointers.get("resume")
    if not isinstance(ledger, Mapping) or not isinstance(resume, Mapping):
        raise PackageContractError("Snapshot pointers are malformed.")
    roles = {
        "checkpoint.json": "checkpoint",
        str(ledger.get("path")): "estimator_ledger_checkpoint",
        str(resume.get("path")): "verified_resume_sidecar",
    }
    if len(roles) != 3 or set(roles) != set(raw_members):
        raise PackageContractError("Snapshot pointer-closed triplet drifted.")
    members: list[dict[str, Any]] = []
    for name, role in roles.items():
        row = raw_members[name]
        if not isinstance(row, Mapping) or not is_hex64(row.get("sha256")):
            raise PackageContractError("Snapshot member binding drifted.")
        members.append(
            {
                "role": role,
                "path": name,
                "sha256": row["sha256"],
                "size_bytes": int(row["size_bytes"]),
            }
        )
    members.sort(key=lambda row: str(row["role"]))
    checkpoint = next(row for row in members if row["role"] == "checkpoint")
    return {
        "local_archive": {
            "path": archive.relative_to(repo_root).as_posix(),
            "sha256": spec["archive_sha256"],
            "size_bytes": spec["archive_size_bytes"],
        },
        "runtime_archive_basename": archive.name,
        "validation": file_binding(validation_path, relative_to=repo_root),
        "authenticated_projection": json_binding(
            projection_path, relative_to=repo_root
        ),
        "checkpoint_path": "checkpoint.json",
        "checkpoint_sha256": checkpoint["sha256"],
        "members": members,
        "member_count": 3,
        "pointer_closed": True,
        "resume_controller_round": spec["resume_controller_round"],
        "source_cluster_id": SOURCE_CLUSTER_ID,
        "source_proc_id": spec["proc_id"],
    }


def expected_materialization(
    repo_root: Path, *, hash_archives: bool
) -> list[dict[str, Any]]:
    _source_manifest_path, source_manifest = _verified_manifest(
        repo_root,
        SOURCE_PACKAGE_RELATIVE,
        file_sha256=SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
        canonical_sha256=SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
        label="source package",
    )
    _loader_manifest_path, loader_manifest = _verified_manifest(
        repo_root,
        LOADER_PACKAGE_RELATIVE,
        file_sha256=LOADER_PACKAGE_MANIFEST_FILE_SHA256,
        canonical_sha256=LOADER_PACKAGE_MANIFEST_CANONICAL_SHA256,
        label="loader package",
    )
    source_archive = repo_root / SOURCE_PACKAGE_RELATIVE / "source/source_locked.tar.gz"
    source_runner = repo_root / SOURCE_PACKAGE_RELATIVE / "run_cell.py"
    loader_runner = repo_root / LOADER_PACKAGE_RELATIVE / "run_resume_cell.py"
    if (
        sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256
        or sha256_file(source_runner) != SOURCE_RUNNER_SHA256
        or sha256_file(loader_runner) != LOADER_RUNNER_SHA256
    ):
        raise PackageContractError("Source/loader runtime bytes drifted.")
    rows: list[dict[str, Any]] = []
    for spec in ROW_SPECS:
        source_job_path, source_job, source_job_binding = _bound_job(
            repo_root=repo_root,
            package_relative=SOURCE_PACKAGE_RELATIVE,
            manifest=source_manifest,
            execution_id_value=str(spec["base_execution_id"]),
            label="source",
        )
        predecessor_path, predecessor, predecessor_binding = _bound_job(
            repo_root=repo_root,
            package_relative=LOADER_PACKAGE_RELATIVE,
            manifest=loader_manifest,
            execution_id_value=str(spec["predecessor_execution_id"]),
            label="predecessor loader",
        )
        source_protocol_path, source_protocol = _source_protocol(
            repo_root, source_job
        )
        if (
            source_job.get("regime_id") != spec["regime_id"]
            or source_job.get("nph") != 7
            or source_job.get("route_contract_sha256")
            != ROUTE_CONTRACT_SHA256
            or predecessor.get("source_execution_id")
            != spec["base_execution_id"]
            or predecessor.get("regime_id") != spec["regime_id"]
            or predecessor.get("implementation_repair")
            != implementation_repair()
            or predecessor.get("resume_input", {}).get(
                "resume_controller_round"
            )
            != spec["source_resume_controller_round"]
        ):
            raise PackageContractError("Source/predecessor row identity drifted.")
        derived = derived_protocol(source_protocol)
        resume = _snapshot_binding(
            repo_root, spec, hash_archive=hash_archives
        )
        rows.append(
            {
                "spec": dict(spec),
                "execution_id": execution_id(spec),
                "source_job_path": source_job_path,
                "source_job": source_job,
                "source_job_binding": source_job_binding,
                "predecessor_path": predecessor_path,
                "predecessor": predecessor,
                "predecessor_binding": predecessor_binding,
                "source_protocol_path": source_protocol_path,
                "source_protocol": source_protocol,
                "derived_protocol": derived,
                "resume_input": resume,
            }
        )
    return rows
