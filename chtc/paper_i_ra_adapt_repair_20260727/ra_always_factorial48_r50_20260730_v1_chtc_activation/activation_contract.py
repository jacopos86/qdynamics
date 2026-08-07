#!/usr/bin/env python3
"""Fail-closed activation contract for the 48-cell RA-always factorial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_always_factorial48_r50_20260730_v1_chtc"
)
ACTIVATION_ID = (
    "paper_i_ra_adapt_always_factorial48_r50_20260730_v1_"
    "chtc_activation_v1"
)
BATCH_NAME = "paper-i-ra-adapt-always-factorial48-r50-20260730-v1"
PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v1_chtc"
)
ACTIVATION_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v1_chtc_activation"
)
RUNTIME_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v1_chtc_runtime"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "9f66b50958001a359229ed2d70c90465f716f239c327a81c987d7fb6b7581092"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "ea2170a68521d01cb2ee865807b635b37365a70c152391d234fe5d48e793253c"
)
EXECUTION_PLAN_CANONICAL_SHA256 = (
    "6c3fa999fb3ed59c8c9d3e07cb51eb73692eaf9a0bc1e01c78866117eae09120"
)
EXECUTION_PLAN_FILE_SHA256 = (
    "68644ee7e7482d922839726242e3e6b250970cd4ccdbfbbd3dbee6c7d4304a5a"
)
SOURCE_ARCHIVE_SHA256 = (
    "efae5d26981bab62cfb2b6dbf077effb4f19a7da6f636dde2b16fbf7acde76b6"
)

JOB_SCHEMA = "paper_i_ra_always_factorial_job_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_always_factorial_execution_authorization_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_always_factorial_activation_manifest_v1"
)
REMOTE_DRY_RUN_SCHEMA = (
    "paper_i_ra_always_factorial_remote_dry_run_validation_v1"
)
DIRECT_EXECUTION_COUNT = 48
MAX_MATERIALIZE = 4
LEAVE_IN_QUEUE = True
QUEUE_VARIABLES = (
    "execution_id",
    "job_path",
    "job_file_sha256",
    "authorization_path",
    "authorization_file_sha256",
    "cpus",
    "memory_mb",
    "disk_mb",
    "max_runtime_seconds",
)
JOB_RESOURCE_KEYS = (
    "request_cpus",
    "request_memory_mb",
    "request_disk_mb",
    "max_runtime_seconds",
)
CONTROL_FILES = (
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "build_attempt_archive.py",
    "execute_authorized_job.sh",
    "submit.sub",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSIGNMENT_RE = re.compile(
    r"^\s*\+?([A-Za-z][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*;?\s*$"
)
_FORBIDDEN_DRY_RUN_PATTERNS = (
    re.compile(r"\bRequestmemory_mb\b", re.IGNORECASE),
    re.compile(r"\bRequestdisk_mb\b", re.IGNORECASE),
    re.compile(r"\bTARGET\.memory_mb\b", re.IGNORECASE),
    re.compile(r"\bTARGET\.disk_mb\b", re.IGNORECASE),
)


class ActivationContractError(ValueError):
    """Raised when activation or remote dry-run state drifts."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"{label} is missing or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationContractError(f"{label} is unreadable.") from exc
    if not isinstance(value, dict):
        raise ActivationContractError(f"{label} is not a mapping.")
    return value


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("sha256") != canonical_sha256(payload):
        raise ActivationContractError(f"{label} self-digest drifted.")


def safe_relative_path(value: Any, *, label: str) -> Path:
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise ActivationContractError(f"Unsafe {label}: {value}")
    return Path(*path.parts)


def repo_root_from_script(script: str | Path) -> Path:
    path = Path(script).resolve()
    for parent in (path.parent, *path.parents):
        if (
            (parent / "AGENTS.md").is_file()
            and (parent / "pipelines").is_dir()
            and (parent / "chtc").is_dir()
        ):
            return parent
    raise ActivationContractError("Repository root could not be resolved.")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"Unsafe binding source: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=f"{path.name} binding")
    verify_self_digest(payload, label=f"{path.name} binding")
    return {
        **file_binding(path, relative_to=relative_to),
        "canonical_sha256": payload["sha256"],
    }


def _verify_file_binding(
    binding: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    json_payload: bool = False,
) -> Path:
    path = base / safe_relative_path(binding.get("path"), label=label)
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != binding.get("sha256")
        or path.stat().st_size != int(binding.get("size_bytes", -1))
    ):
        raise ActivationContractError(f"{label} file binding drifted.")
    if json_payload:
        payload = load_json(path, label=label)
        verify_self_digest(payload, label=label)
        if payload["sha256"] != binding.get("canonical_sha256"):
            raise ActivationContractError(
                f"{label} canonical binding drifted."
            )
    return path


def package_inventory(package_dir: Path) -> tuple[str, ...]:
    if not package_dir.is_dir() or package_dir.is_symlink():
        raise ActivationContractError("Factorial package is unavailable.")
    paths = tuple(package_dir.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ActivationContractError("Factorial package contains a symlink.")
    return tuple(
        sorted(
            path.relative_to(package_dir).as_posix()
            for path in paths
            if path.is_file()
        )
    )


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ActivationContractError(f"{label} is not a SHA-256 digest.")
    return text


def _validated_job_resources(job: Mapping[str, Any]) -> dict[str, int]:
    raw = job.get("resources")
    if (
        not isinstance(raw, Mapping)
        or set(raw) != set(JOB_RESOURCE_KEYS)
    ):
        raise ActivationContractError("Job resource contract drifted.")
    resources: dict[str, int] = {}
    for name in JOB_RESOURCE_KEYS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ActivationContractError(
                f"Job resource {name} must be a positive integer."
            )
        resources[name] = int(value)
    return resources


def validate_authorization_payload(
    authorization: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    verify_self_digest(authorization, label="execution authorization")
    execution_id = execution.get("execution_id")
    remote_image = manifest.get("remote_image")
    if not isinstance(remote_image, Mapping):
        raise ActivationContractError(
            "Activation remote-image binding is absent."
        )
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_id": f"{ACTIVATION_ID}__{execution_id}",
        "authorized_utc": manifest.get("authorized_utc"),
        "package_id": PACKAGE_ID,
        "activation_id": ACTIVATION_ID,
        "batch_name": BATCH_NAME,
        "execution_id": execution_id,
        "job_sha256": execution.get("job", {}).get("canonical_sha256"),
        "job_file_sha256": execution.get("job", {}).get("sha256"),
        "package_manifest_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
        "execution_plan_sha256": EXECUTION_PLAN_CANONICAL_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "activation_control_plane_sha256": manifest.get(
            "activation_control_plane_sha256"
        ),
        "remote_image_path": REMOTE_IMAGE_PATH,
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "remote_image_byte_verification_passed": True,
        "remote_image_verified_utc": remote_image.get("verified_utc"),
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ActivationContractError("Execution authorization binding drifted.")
    if not authorization.get("authorized_utc") or not authorization.get(
        "remote_image_verified_utc"
    ):
        raise ActivationContractError("Execution authorization is undated.")


def _parse_queue(path: Path) -> list[list[str]]:
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != DIRECT_EXECUTION_COUNT or any(
        len(row) != len(QUEUE_VARIABLES) for row in rows
    ):
        raise ActivationContractError("Activation queue is malformed.")
    return rows


def _submit_queue_variables(submit: str) -> tuple[str, ...]:
    matches = re.findall(
        r"(?im)^\s*queue\s+(.+?)\s+from\s+\S+\s*$",
        submit,
    )
    if len(matches) != 1:
        raise ActivationContractError(
            "Submit description must have exactly one queue-from statement."
        )
    variables = tuple(
        token.strip() for token in matches[0].split(",") if token.strip()
    )
    if variables != QUEUE_VARIABLES:
        raise ActivationContractError("Submit queue variables drifted.")
    if any(name.lower().startswith("request_") for name in variables):
        raise ActivationContractError(
            "Submit queue variables may not shadow request attributes."
        )
    return variables


def _submit_assignment_values(
    submit: str,
    name: str,
) -> tuple[str, ...]:
    pattern = re.compile(
        rf"(?im)^\s*\+?{re.escape(name)}\s*=\s*(.*?)\s*$"
    )
    return tuple(match.group(1).strip() for match in pattern.finditer(submit))


def _validate_factory_submit_policy(submit: str) -> None:
    if _submit_assignment_values(
        submit, "max_materialize"
    ) != (str(MAX_MATERIALIZE),):
        raise ActivationContractError(
            "Submit description must assign the exact factory "
            "materialization limit once."
        )
    if _submit_assignment_values(
        submit, "leave_in_queue"
    ) != ("True",):
        raise ActivationContractError(
            "Submit description must retain completed jobs exactly once."
        )
    for competing_name in (
        "max_idle",
        "JobMaterializeLimit",
        "JobMaterializeMaxIdle",
        "MY.JobMaterializeLimit",
        "MY.JobMaterializeMaxIdle",
    ):
        if _submit_assignment_values(submit, competing_name):
            raise ActivationContractError(
                "Submit description contains a competing factory control."
            )


def validate_submit_text(submit: str) -> None:
    _submit_queue_variables(submit)
    _validate_factory_submit_policy(submit)
    lowered = submit.lower()
    if "$(request_" in lowered:
        raise ActivationContractError(
            "Submit description uses a request_-prefixed item variable."
        )
    expected_lines = {
        "request_cpus = $(cpus)",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        "+MaxRuntime = $(max_runtime_seconds)",
        "when_to_transfer_output = ON_EXIT",
        "requirements = TARGET.HasSIF",
        f'+JobBatchName = "{BATCH_NAME}"',
        f"max_materialize = {MAX_MATERIALIZE}",
        "leave_in_queue = True",
    }
    observed_lines = {
        line.strip() for line in submit.splitlines() if line.strip()
    }
    if not expected_lines.issubset(observed_lines):
        raise ActivationContractError(
            "Submit resource or lifecycle assignments drifted."
        )
    if "ON_EXIT_OR_EVICT" in submit:
        raise ActivationContractError("Eviction output transfer is forbidden.")
    if any(pattern.search(submit) for pattern in _FORBIDDEN_DRY_RUN_PATTERNS):
        raise ActivationContractError(
            "Submit description uses a nonexistent resource attribute."
        )
    if SOURCE_ARCHIVE_SHA256 not in submit:
        raise ActivationContractError(
            "Submit description lost its source-archive digest."
        )
    if (
        REMOTE_IMAGE_PATH not in submit
        or REMOTE_IMAGE_SHA256 not in submit
    ):
        raise ActivationContractError(
            "Submit description lost its remote-image binding."
        )
    package_text = PACKAGE_RELATIVE.as_posix()
    activation_text = ACTIVATION_RELATIVE.as_posix()
    runtime_text = RUNTIME_RELATIVE.as_posix()
    for required in (package_text, activation_text, runtime_text):
        if required not in submit:
            raise ActivationContractError(
                f"Submit description lost path binding: {required}"
            )
    archive_name = (
        "$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    )
    fetched_name = (
        "$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"
    )
    if archive_name not in submit or fetched_name not in submit:
        raise ActivationContractError(
            "Submit description lost unique attempt-output naming."
        )


def expanded_dry_run_submit_text(submit: str) -> str:
    """Return the sealed nonfactory projection used to expand every row."""
    validate_submit_text(submit)
    pattern = re.compile(r"(?i)^\s*max_materialize\s*=")
    projected_lines: list[str] = []
    removed = 0
    for line in submit.splitlines(keepends=True):
        if pattern.match(line):
            removed += 1
        else:
            projected_lines.append(line)
    if removed != 1:
        raise ActivationContractError(
            "Expanded dry-run projection did not remove exactly one "
            "factory limit."
        )
    projected = "".join(projected_lines)
    if _submit_assignment_values(projected, "max_materialize"):
        raise ActivationContractError(
            "Expanded dry-run projection retained a factory limit."
        )
    return projected


def remote_dry_run_contract() -> dict[str, Any]:
    return {
        "schema": REMOTE_DRY_RUN_SCHEMA,
        "required_before_condor_submit": True,
        "factory_cluster_dry_run": {
            "expected_cluster_ad_count": 1,
            "proc_id_forbidden": True,
            "required_observed_attributes": [
                "ClusterId",
                "JobBatchName",
                "LeaveJobInQueue",
            ],
        },
        "expanded_nonfactory_projection": {
            "transform": "remove_exact_max_materialize_assignment_v1",
            "expected_ad_count": DIRECT_EXECUTION_COUNT,
            "expected_proc_ids": list(range(DIRECT_EXECUTION_COUNT)),
            "required_resource_attributes": [
                "RequestCpus",
                "RequestMemory",
                "RequestDisk",
                "MaxRuntime",
            ],
            "required_lifecycle_attributes": ["LeaveJobInQueue"],
            "request_memory_unit": "MiB",
            "request_disk_unit": "KiB",
            "request_disk_conversion": (
                "job_request_disk_mb_times_1024_v1"
            ),
        },
        "post_submit_factory_query": {
            "required": True,
            "observed_in_pre_submit_dry_run": False,
            "expected_attributes": {
                "JobMaterializeLimit": MAX_MATERIALIZE,
                "TotalSubmitProcs": DIRECT_EXECUTION_COUNT,
            },
        },
        "forbidden_attributes": [
            "Requestmemory_mb",
            "Requestdisk_mb",
            "TARGET.memory_mb",
            "TARGET.disk_mb",
        ],
    }


def _parse_classad_blocks(text: str) -> list[dict[str, str]]:
    bracketed = [
        match.group(1)
        for match in re.finditer(r"\[(.*?)\]", text, re.DOTALL)
        if re.search(r"(?im)^\s*ProcId\s*=", match.group(1))
    ]
    if bracketed:
        raw_blocks = bracketed
    else:
        raw_blocks = [
            block
            for block in re.split(r"\n\s*\n", text)
            if re.search(r"(?im)^\s*ProcId\s*=", block)
        ]
    ads: list[dict[str, str]] = []
    baseline: dict[str, str] = {}
    inherited_names = {
        "requestcpus",
        "requestmemory",
        "requestdisk",
        "maxruntime",
        "jobbatchname",
        "leavejobinqueue",
    }
    for block in raw_blocks:
        ad: dict[str, str] = {}
        for line in block.splitlines():
            match = _ASSIGNMENT_RE.match(line)
            if match is None:
                continue
            name, value = match.groups()
            lowered = name.lower()
            if lowered in ad:
                raise ActivationContractError(
                    f"Remote dry-run ad repeats attribute {name}."
                )
            ad[lowered] = value.strip().rstrip(";").strip()
        if "procid" in ad:
            if bracketed:
                ads.append(ad)
            else:
                # HTCondor 25.13 emits the first dry-run ad in full and
                # subsequent ads as deltas against that cluster baseline.
                # Each process overlays the baseline independently.
                if not baseline:
                    baseline = {
                        name: value
                        for name, value in ad.items()
                        if name in inherited_names
                    }
                    ads.append(ad)
                else:
                    ads.append({**baseline, **ad})
    return ads


def _integer_ad_value(ad: Mapping[str, str], name: str) -> int:
    raw = ad.get(name.lower())
    if raw is None:
        raise ActivationContractError(
            f"Remote dry-run ad is missing {name}."
        )
    unquoted = raw.strip().strip('"')
    if re.fullmatch(r"[0-9]+", unquoted) is None:
        raise ActivationContractError(
            f"Remote dry-run {name} is not an integer."
        )
    return int(unquoted)


def _string_ad_value(ad: Mapping[str, str], name: str) -> str:
    raw = ad.get(name.lower())
    if raw is None:
        raise ActivationContractError(
            f"Remote dry-run ad is missing {name}."
        )
    return raw.strip().strip('"')


def _boolean_ad_value(ad: Mapping[str, str], name: str) -> bool:
    value = _string_ad_value(ad, name).lower()
    if value not in {"true", "false"}:
        raise ActivationContractError(
            f"Remote dry-run {name} is not a boolean."
        )
    return value == "true"


def _parse_factory_cluster_ad(text: str) -> dict[str, str]:
    if re.search(r"(?im)^\s*\+?ProcId\s*=", text):
        raise ActivationContractError(
            "Factory dry-run unexpectedly materialized a ProcId."
        )
    bracketed = [
        match.group(1)
        for match in re.finditer(r"\[(.*?)\]", text, re.DOTALL)
        if re.search(
            r"(?im)^\s*ClusterId\s*=",
            match.group(1),
        )
    ]
    raw_blocks = (
        bracketed
        if bracketed
        else [
            block
            for block in re.split(r"\n\s*\n", text)
            if block.strip()
        ]
    )
    ads: list[dict[str, str]] = []
    for block in raw_blocks:
        ad: dict[str, str] = {}
        for line in block.splitlines():
            match = _ASSIGNMENT_RE.match(line)
            if match is None:
                continue
            name, value = match.groups()
            lowered = name.lower()
            if lowered in ad:
                raise ActivationContractError(
                    f"Factory dry-run repeats attribute {name}."
                )
            ad[lowered] = value.strip().rstrip(";").strip()
        if ad:
            ads.append(ad)
    if len(ads) != 1:
        raise ActivationContractError(
            "Factory dry-run must contain exactly one cluster ad."
        )
    return ads[0]


def validate_remote_factory_dry_run_text(text: str) -> dict[str, Any]:
    if any(pattern.search(text) for pattern in _FORBIDDEN_DRY_RUN_PATTERNS):
        raise ActivationContractError(
            "Factory dry-run contains a nonexistent resource attribute."
        )
    ad = _parse_factory_cluster_ad(text)
    cluster_id = _integer_ad_value(ad, "ClusterId")
    batch_name = _string_ad_value(ad, "JobBatchName")
    leave_in_queue = _boolean_ad_value(ad, "LeaveJobInQueue")
    if batch_name != BATCH_NAME:
        raise ActivationContractError(
            "Factory dry-run batch identity drifted."
        )
    if leave_in_queue is not LEAVE_IN_QUEUE:
        raise ActivationContractError(
            "Factory dry-run completion-retention policy drifted."
        )
    return digested(
        {
            "schema": REMOTE_DRY_RUN_SCHEMA,
            "dry_run_kind": "factory_cluster_ad_v1",
            "status": "passed",
            "cluster_id": cluster_id,
            "batch_name": batch_name,
            "observed_leave_in_queue": leave_in_queue,
            "classad_text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "live_factory_query_required": True,
            "live_factory_expected_attributes": {
                "JobMaterializeLimit": MAX_MATERIALIZE,
                "TotalSubmitProcs": DIRECT_EXECUTION_COUNT,
            },
        }
    )


def validate_remote_dry_run_text(
    text: str,
    *,
    executions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if any(pattern.search(text) for pattern in _FORBIDDEN_DRY_RUN_PATTERNS):
        raise ActivationContractError(
            "Remote dry-run materialized a nonexistent resource attribute."
        )
    ads = _parse_classad_blocks(text)
    if len(ads) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError(
            "Remote dry-run must materialize exactly 48 ads."
        )
    if len(executions) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError(
            "Activation execution count drifted before dry-run validation."
        )

    rows: list[dict[str, Any]] = []
    observed_proc_ids: list[int] = []
    for index, (ad, execution) in enumerate(
        zip(ads, executions)
    ):
        proc_id = _integer_ad_value(ad, "ProcId")
        observed_proc_ids.append(proc_id)
        if proc_id != index:
            raise ActivationContractError(
                "Remote dry-run ProcId order is not exactly 0..47."
            )
        resources = execution.get("resources")
        if not isinstance(resources, Mapping):
            raise ActivationContractError(
                "Activation execution has no resource contract."
            )
        expected = {
            "RequestCpus": int(resources["request_cpus"]),
            "RequestMemory": int(resources["request_memory_mb"]),
            "RequestDisk": int(resources["request_disk_mb"]) * 1024,
            "MaxRuntime": int(resources["max_runtime_seconds"]),
        }
        observed = {
            name: _integer_ad_value(ad, name) for name in expected
        }
        if observed != expected:
            raise ActivationContractError(
                f"Remote dry-run resource drift for ProcId {proc_id}."
            )
        if _string_ad_value(ad, "JobBatchName") != BATCH_NAME:
            raise ActivationContractError(
                "Remote dry-run batch identity drifted."
            )
        if _boolean_ad_value(ad, "LeaveJobInQueue") is not LEAVE_IN_QUEUE:
            raise ActivationContractError(
                "Remote dry-run completion-retention policy drifted."
            )
        rows.append(
            {
                "proc_id": proc_id,
                "execution_id": execution["execution_id"],
                **observed,
            }
        )
    if observed_proc_ids != list(range(DIRECT_EXECUTION_COUNT)):
        raise ActivationContractError(
            "Remote dry-run ProcId closure drifted."
        )
    return digested(
        {
            "schema": REMOTE_DRY_RUN_SCHEMA,
            "dry_run_kind": "expanded_nonfactory_projection_v1",
            "status": "passed",
            "ad_count": len(ads),
            "proc_ids": observed_proc_ids,
            "batch_name": BATCH_NAME,
            "observed_leave_in_queue": LEAVE_IN_QUEUE,
            "classad_text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "resources": rows,
        }
    )


def validate_remote_dry_run(
    repo_root: str | Path,
    dry_run_path: str | Path,
) -> dict[str, Any]:
    result = validate_activation(repo_root)
    path = Path(dry_run_path)
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(
            "Remote dry-run classad file is missing or unsafe."
        )
    return validate_remote_dry_run_text(
        path.read_text(encoding="utf-8"),
        executions=result["manifest"]["executions"],
    )


def validate_remote_factory_dry_run(
    repo_root: str | Path,
    dry_run_path: str | Path,
) -> dict[str, Any]:
    validate_activation(repo_root)
    path = Path(dry_run_path)
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(
            "Remote factory dry-run classad file is missing or unsafe."
        )
    return validate_remote_factory_dry_run_text(
        path.read_text(encoding="utf-8")
    )


def validate_activation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package_dir = root / PACKAGE_RELATIVE
    activation_dir = root / ACTIVATION_RELATIVE
    observed_package_inventory = set(package_inventory(package_dir))

    package_manifest_path = package_dir / "package_manifest.json"
    package_manifest = load_json(
        package_manifest_path, label="factorial package manifest"
    )
    verify_self_digest(package_manifest, label="factorial package manifest")
    plan_path = package_dir / "execution_plan.json"
    plan = load_json(plan_path, label="factorial execution plan")
    verify_self_digest(plan, label="factorial execution plan")
    if (
        sha256_file(package_manifest_path)
        != PACKAGE_MANIFEST_FILE_SHA256
        or package_manifest.get("sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or sha256_file(plan_path) != EXECUTION_PLAN_FILE_SHA256
        or plan.get("sha256") != EXECUTION_PLAN_CANONICAL_SHA256
        or sha256_file(package_dir / "source_locked.tar.gz")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise ActivationContractError("Factorial package authority drifted.")
    if (
        package_manifest.get("package_id") != PACKAGE_ID
        or plan.get("package_id") != PACKAGE_ID
        or int(package_manifest.get("direct_execution_count", -1))
        != DIRECT_EXECUTION_COUNT
        or int(plan.get("direct_execution_count", -1))
        != DIRECT_EXECUTION_COUNT
    ):
        raise ActivationContractError("Factorial package identity drifted.")

    package_controls = package_manifest.get("control_plane")
    source_archive = package_manifest.get("source_archive")
    source_manifest = (
        source_archive.get("manifest")
        if isinstance(source_archive, Mapping)
        else None
    )
    package_plan = package_manifest.get("execution_plan")
    smoke_receipt = package_manifest.get("smoke_receipt")
    package_queue = package_manifest.get("queue")
    package_jobs = package_manifest.get("jobs")
    if (
        not isinstance(package_controls, list)
        or not package_controls
        or not isinstance(source_archive, Mapping)
        or not isinstance(source_manifest, Mapping)
        or not isinstance(package_plan, Mapping)
        or not isinstance(smoke_receipt, Mapping)
        or not isinstance(package_queue, Mapping)
        or not isinstance(package_jobs, list)
    ):
        raise ActivationContractError(
            "Factorial package file bindings are incomplete."
        )
    package_bound_files = {
        "package_manifest.json",
        "execution_plan.json",
    }
    for index, binding in enumerate(package_controls):
        if not isinstance(binding, Mapping):
            raise ActivationContractError(
                "Factorial package control binding is malformed."
            )
        path = _verify_file_binding(
            binding,
            base=package_dir,
            label=f"factorial package control {index}",
        )
        package_bound_files.add(
            path.relative_to(package_dir).as_posix()
        )
    for label, binding, json_payload in (
        ("source archive", source_archive, False),
        ("source archive manifest", source_manifest, True),
        ("execution plan", package_plan, True),
        ("smoke receipt", smoke_receipt, True),
        ("package queue", package_queue, False),
    ):
        path = _verify_file_binding(
            binding,
            base=package_dir,
            label=label,
            json_payload=json_payload,
        )
        package_bound_files.add(
            path.relative_to(package_dir).as_posix()
        )
    for index, binding in enumerate(package_jobs):
        if not isinstance(binding, Mapping):
            raise ActivationContractError(
                "Factorial package job binding is malformed."
            )
        package_bound_files.add(
            safe_relative_path(
                binding.get("path"),
                label=f"factorial package job {index}",
            ).as_posix()
        )
    if observed_package_inventory != package_bound_files:
        raise ActivationContractError(
            "Factorial package recursive closure drifted."
        )

    manifest_path = activation_dir / "activation_manifest.json"
    manifest = load_json(manifest_path, label="activation manifest")
    verify_self_digest(manifest, label="activation manifest")
    fixed = {
        "schema": ACTIVATION_SCHEMA,
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "batch_name": BATCH_NAME,
        "campaign_id": package_manifest.get("campaign_id"),
        "run_class": package_manifest.get("run_class"),
        "execution_target": "chtc",
        "direct_execution_count": DIRECT_EXECUTION_COUNT,
        "queue_variables": list(QUEUE_VARIABLES),
        "remote_dry_run_validation_contract": remote_dry_run_contract(),
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
        "paper_evidence_adopted": False,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ActivationContractError("Activation state drifted.")
    if not manifest.get("authorized_utc"):
        raise ActivationContractError("Activation authorization is undated.")
    if "cluster_id" in manifest:
        raise ActivationContractError(
            "Pre-submit activation claims a scheduler cluster."
        )
    remote_image = manifest.get("remote_image")
    if (
        not isinstance(remote_image, Mapping)
        or remote_image.get("path") != REMOTE_IMAGE_PATH
        or remote_image.get("sha256") != REMOTE_IMAGE_SHA256
        or remote_image.get("byte_verification_passed") is not True
        or not remote_image.get("verified_utc")
    ):
        raise ActivationContractError("Remote-image binding drifted.")

    package_binding = manifest.get("sealed_package")
    if not isinstance(package_binding, Mapping):
        raise ActivationContractError("Factorial package binding is absent.")
    if package_binding != {
        "path": PACKAGE_RELATIVE.as_posix(),
        "manifest_canonical_sha256": (
            PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        "manifest_file_sha256": PACKAGE_MANIFEST_FILE_SHA256,
        "execution_plan_canonical_sha256": (
            EXECUTION_PLAN_CANONICAL_SHA256
        ),
        "execution_plan_file_sha256": EXECUTION_PLAN_FILE_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }:
        raise ActivationContractError(
            "Activation lost exact factorial-package binding."
        )

    controls = manifest.get("control_plane")
    if not isinstance(controls, list) or [
        row.get("path") for row in controls if isinstance(row, Mapping)
    ] != list(CONTROL_FILES):
        raise ActivationContractError("Activation control plane drifted.")
    for row in controls:
        if not isinstance(row, Mapping):
            raise ActivationContractError("Control-plane row is malformed.")
        _verify_file_binding(
            row, base=activation_dir, label=f"control {row.get('path')}"
        )
    observed_control_digest = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    if observed_control_digest != manifest.get(
        "activation_control_plane_sha256"
    ):
        raise ActivationContractError("Control-plane digest drifted.")

    submit = (activation_dir / "submit.sub").read_text(encoding="utf-8")
    validate_submit_text(submit)

    queue_binding = manifest.get("queue")
    if not isinstance(queue_binding, Mapping):
        raise ActivationContractError("Activation queue binding is absent.")
    queue_path = _verify_file_binding(
        queue_binding, base=activation_dir, label="activation queue"
    )
    queue_rows = _parse_queue(queue_path)

    executions = manifest.get("executions")
    if (
        not isinstance(executions, list)
        or len(executions) != DIRECT_EXECUTION_COUNT
        or len(package_jobs) != DIRECT_EXECUTION_COUNT
    ):
        raise ActivationContractError("Activation execution count drifted.")
    plan_ids = plan.get("execution_ids")
    if not isinstance(plan_ids, list) or len(plan_ids) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError("Execution plan ID closure drifted.")

    authorizations: list[dict[str, Any]] = []
    expected_files = {
        *(activation_dir / name for name in CONTROL_FILES),
        activation_dir / "activation_manifest.json",
        activation_dir / "queue.tsv",
    }
    observed_ids: list[str] = []
    for index, (execution, package_job, queue_row) in enumerate(
        zip(executions, package_jobs, queue_rows)
    ):
        if not isinstance(execution, Mapping) or not isinstance(
            package_job, Mapping
        ):
            raise ActivationContractError("Activation execution is malformed.")
        execution_id = str(execution.get("execution_id"))
        observed_ids.append(execution_id)
        if execution_id != str(plan_ids[index]):
            raise ActivationContractError("Activation execution order drifted.")
        job_binding = execution.get("job")
        authorization_binding = execution.get("authorization")
        if not isinstance(job_binding, Mapping) or not isinstance(
            authorization_binding, Mapping
        ):
            raise ActivationContractError("Execution binding is incomplete.")
        job_path = _verify_file_binding(
            job_binding,
            base=root,
            label=f"{execution_id} job",
            json_payload=True,
        )
        job = load_json(job_path, label=f"{execution_id} job")
        resources = _validated_job_resources(job)
        package_job_path = package_job.get("path")
        if (
            job.get("schema") != JOB_SCHEMA
            or job.get("package_id") != PACKAGE_ID
            or job.get("execution_id") != execution_id
            or job.get("sha256") != job_binding.get("canonical_sha256")
            or package_job_path
            != job_path.relative_to(package_dir).as_posix()
            or package_job.get("canonical_sha256") != job.get("sha256")
            or package_job.get("sha256") != job_binding.get("sha256")
            or execution.get("resources") != resources
        ):
            raise ActivationContractError("Activation job binding drifted.")

        authorization_path = _verify_file_binding(
            authorization_binding,
            base=activation_dir,
            label=f"{execution_id} authorization",
            json_payload=True,
        )
        if authorization_path.relative_to(
            activation_dir
        ).as_posix() != f"authorizations/{execution_id}.json":
            raise ActivationContractError(
                "Execution authorization path drifted."
            )
        authorization = load_json(
            authorization_path, label=f"{execution_id} authorization"
        )
        validate_authorization_payload(
            authorization, execution=execution, manifest=manifest
        )
        authorizations.append(authorization)
        expected_files.add(authorization_path)
        expected_queue = [
            execution_id,
            job_path.relative_to(root).as_posix(),
            str(job_binding["sha256"]),
            authorization_path.relative_to(root).as_posix(),
            str(authorization_binding["sha256"]),
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        ]
        if queue_row != expected_queue or index != int(
            execution.get("queue_index", -1)
        ):
            raise ActivationContractError("Activation queue binding drifted.")

    if len(set(observed_ids)) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError("Activation execution IDs collide.")
    observed_files = {
        path for path in activation_dir.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ActivationContractError("Activation recursive closure drifted.")
    if any(path.is_symlink() for path in activation_dir.rglob("*")):
        raise ActivationContractError("Activation contains a symlink.")
    return {
        "manifest": manifest,
        "package_manifest": package_manifest,
        "execution_plan": plan,
        "authorizations": authorizations,
    }


__all__ = [
    "ACTIVATION_ID",
    "ACTIVATION_RELATIVE",
    "ACTIVATION_SCHEMA",
    "ActivationContractError",
    "AUTHORIZATION_SCHEMA",
    "BATCH_NAME",
    "CONTROL_FILES",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_PLAN_CANONICAL_SHA256",
    "EXECUTION_PLAN_FILE_SHA256",
    "JOB_RESOURCE_KEYS",
    "JOB_SCHEMA",
    "LEAVE_IN_QUEUE",
    "MAX_MATERIALIZE",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_CANONICAL_SHA256",
    "PACKAGE_MANIFEST_FILE_SHA256",
    "PACKAGE_RELATIVE",
    "QUEUE_VARIABLES",
    "REMOTE_DRY_RUN_SCHEMA",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "RUNTIME_RELATIVE",
    "SOURCE_ARCHIVE_SHA256",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "expanded_dry_run_submit_text",
    "file_binding",
    "json_binding",
    "load_json",
    "remote_dry_run_contract",
    "repo_root_from_script",
    "sha256_file",
    "validate_activation",
    "validate_authorization_payload",
    "validate_remote_factory_dry_run",
    "validate_remote_factory_dry_run_text",
    "validate_remote_dry_run",
    "validate_remote_dry_run_text",
    "validate_submit_text",
    "verify_self_digest",
]
