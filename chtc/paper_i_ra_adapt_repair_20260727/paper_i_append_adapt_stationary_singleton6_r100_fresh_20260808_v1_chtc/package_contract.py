#!/usr/bin/env python3
"""Fail-closed contract for the inert conventional-Append round-100 package."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_ID = (
    "paper_i_append_adapt_stationary_singleton6_r100_fresh_"
    "20260808_v1_chtc"
)
CAMPAIGN_ID = (
    "paper_i_append_adapt_stationary_singleton6_r100_fresh_v1"
)
RUN_CLASS = "paper_facing"
EXECUTION_TARGET = "chtc"
SOURCE_HORIZON = 50
TARGET_HORIZON = 100
DIRECT_EXECUTION_COUNT = 6

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/" + PACKAGE_ID
)

SOURCE_PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6_chtc"
)
SOURCE_PACKAGE_DIRECTORY_NAME = (
    "stationary_core_full48_r50_20260728_v6_chtc"
)
SOURCE_PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    + SOURCE_PACKAGE_DIRECTORY_NAME
)
SOURCE_PACKAGE_DIR = REPO_ROOT / SOURCE_PACKAGE_RELATIVE_ROOT
SOURCE_BUNDLE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v10/"
    "ra_repair_stationary_late_core_v1"
)
SOURCE_ARCHIVE_NAME = "source_r50_locked.tar.gz"
SOURCE_ARCHIVE_MANIFEST_NAME = "source_r50_archive_manifest.json"
SOURCE_ARCHIVE_SHA256 = (
    "1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee"
)
SOURCE_ARCHIVE_SIZE_BYTES = 2_418_129
SOURCE_PACKAGE_MANIFEST_SHA256 = (
    "75063a0d8de86518d91a55283e025037229d20c185681db74b79175f9b9e6176"
)
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "48a32ad64773a794f748b4c4e6013cf52e7d637a0c8f8c7da83b71a07df280cc"
)
SOURCE_EXECUTION_PLAN_SHA256 = (
    "d8f52fbedf4cce26caf1407a33d6dd62bd5a732546122e20a4d34a1d449ed6ad"
)
SOURCE_EXECUTION_PLAN_FILE_SHA256 = (
    "e1c5c9f550a4038fef55c1ec095c9b5286205432ffc510635f4598ede44c6ce3"
)
SOURCE_ARCHIVE_MANIFEST_SHA256 = (
    "c843a1332026ce8d3236a59adcd53e29770dad63a535ed8fa09990313ed80b72"
)
SOURCE_ARCHIVE_MANIFEST_FILE_SHA256 = (
    "fd1d0f0046bbf6d2cdbba7c718a451351eda46b8802cdb16666deba43bfbf029"
)
SOURCE_FINAL_RECEIPT_SHA256 = (
    "5924cf714ca3f1a36b3b766b4c5e30c5599d1606267ffd98fc08685afa1a9e80"
)
SOURCE_FINAL_RECEIPT_FILE_SHA256 = (
    "8c396883ebd728150057eb9b223793621f774b0475fde59a80675d2de2ccd354"
)
SOURCE_AUTHORIZATION_SHA256 = (
    "540daa954160468cd5dbfea97ae8ab2034c700fc9ba1a2eccd3abd7f20bf69fc"
)
SOURCE_AUTHORIZATION_FILE_SHA256 = (
    "a3a5a24634a0384cffa703b6fa94ad24a5f07672047ef1d222fe34660527ebfc"
)

ANCHOR_EVIDENCE_NAME = "anchor_evidence.json"
ANCHOR_EVIDENCE_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_independent_anchor_v1"
)
ANCHOR_EXECUTION_ID = "core__weak_weak__nph3__append_singleton"
ANCHOR_PROTOCOL_SHA256 = (
    "903377134094eedb21bfd883e2e7041a9aac2d8950302b3a0926cc95c9e4d677"
)
ANCHOR_SOURCE_ARCHIVE_SHA256 = SOURCE_ARCHIVE_SHA256
ANCHOR_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ANCHOR_RESULT_SHA256 = (
    "3c43d7766bfeea4ea30d95b05ccf8d29f1be2d258f778fd38bf2d71971cf0df3"
)
ANCHOR_RESULT_SIZE_BYTES = 441_780_454
ANCHOR_SUMMARY_SHA256 = (
    "090c36168f001d22a81a50262a9772c00dd4ca76eec686aa9dd7916e4f63ea11"
)
ANCHOR_SUMMARY_SIZE_BYTES = 71_162
ANCHOR_CHECKPOINT_SHA256 = (
    "4d1fdfc8f4cc9b311a24d649c20d5b6eadae6dd9d8ab7d0998af6caf9d927826"
)
ANCHOR_CHECKPOINT_SIZE_BYTES = 416_550
ANCHOR_ATTEMPT_RELATIVE_PATH = (
    "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v5_"
    "9392023_20260729/core__weak_weak__nph3__append_singleton__"
    "cluster_9392023__proc_4.tar.gz"
)
ANCHOR_ATTEMPT_SHA256 = (
    "7b0183478c04e5874af83f5cc3fde66d6105708993b58814d7c4c154576b24d8"
)
ANCHOR_ATTEMPT_SIZE_BYTES = 177_074_477
ANCHOR_PACKAGE_DIRECTORY_NAME = (
    "stationary_core_full48_r50_20260728_v5_chtc"
)
ANCHOR_PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v5_chtc"
)
ANCHOR_CLUSTER_ID = 9_392_023
ANCHOR_PROC_ID = 4
ANCHOR_COMPLETED_UTC = "2026-07-29T04:36:09.816640+00:00"
ANCHOR_JOB_CANONICAL_SHA256 = (
    "496a06431bd54d465ce6fb7e133e263452678d5f554288bca1c96749ca6d8f38"
)
ANCHOR_JOB_FILE_SHA256 = (
    "32409870d6b94dc6efcf6beeae42b99a1f40415464c43f9eaa8ba041b51c8ca7"
)
ANCHOR_JOB_SIZE_BYTES = 3_783
ANCHOR_RUN_CELL_SHA256 = (
    "28ba83f6e92e63237159212f977b67a5a66a65a51574d95d4505a2ffc5785cc6"
)
ANCHOR_EXECUTION_MANIFEST_SHA256 = (
    "1f5ae085bfd6b61e538a3785abfce836fb573246b973b9920595c2a3f5c480c2"
)
ANCHOR_EXECUTION_MANIFEST_SIZE_BYTES = 1_389
ANCHOR_WORKER_EXIT_STATUS = 2
ANCHOR_WORKER_EXIT_SHA256 = (
    "53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3"
)
SOURCE_ATTEMPT_RELATIVE_PATH = (
    "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v6_"
    "9392337_20260729/core__weak_weak__nph3__append_singleton__"
    "cluster_9392337__proc_4.tar.gz"
)
SOURCE_ATTEMPT_SHA256 = (
    "3f8d07ea935f156de03490c94f4e85f794f0809511e56f3f37e56b249d825490"
)
SOURCE_ATTEMPT_SIZE_BYTES = 177_079_366
SOURCE_CLUSTER_ID = 9_392_337
SOURCE_PROC_ID = 4
SOURCE_COMPLETED_UTC = "2026-07-29T08:55:05.600139+00:00"
SOURCE_ANCHOR_JOB_CANONICAL_SHA256 = (
    "aab873cdcde816ebdfb8ded6c687355052fe7362c1bd52492f5c7c0e475b3d89"
)
SOURCE_ANCHOR_JOB_FILE_SHA256 = (
    "60a6b73ef5a3b2549841c67237f51e088ab878f3272da4bfd4ee0df4d60b7a26"
)
SOURCE_ANCHOR_JOB_SIZE_BYTES = 3_783
SOURCE_RUN_CELL_SHA256 = (
    "e13446300ae1d3d636eb21df1c36054181f3c95f81d44cea9c9609cd8aa7b0f6"
)
SOURCE_EXECUTION_MANIFEST_SHA256 = (
    "d7d5e21b3e7e17215273409e67168e7eb93e1ae0931c392c15b6f786528220d9"
)
SOURCE_EXECUTION_MANIFEST_SIZE_BYTES = 1_389
SOURCE_WORKER_EXIT_STATUS = 0
SOURCE_WORKER_EXIT_SHA256 = (
    "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
)
SOURCE_WORKER_RECEIPT_SHA256 = (
    "a80aa2e6a67ece5ff5250d173bd9e806df130ca6c3dae20d8da3276a85a4be56"
)
SOURCE_WORKER_RECEIPT_SIZE_BYTES = 20_771
ANCHOR_EXECUTE_WRAPPER_SHA256 = (
    "128163879cfe55071182fc0f1a377984eef7db5c0b1c068e2ecfd43f3a7a7a27"
)
ANCHOR_NORMALIZED_SCIENTIFIC_JOB_SHA256 = (
    "cd374766c6d6dff9a457ca5b12949903b9077f53a7452373f6f10dc13727ea26"
)
ANCHOR_PRIMARY_INVOKE_AST_SHA256 = (
    "b9afa6a8ccd0d0cc99af49bde6eb898d213ebab219624e70d585f140146714d0"
)
ANCHOR_NON_FUNCTION_AST_SHA256 = (
    "6e7b7bc631b91efb0916d7f5472e4fcbe44f08229328f31085f480cacb392ed3"
)
ANCHOR_OPERATOR_SEQUENCE_SHA256 = (
    "71a21e225b0325b3648090f19c96d3cd27150f6ea55585f8f9d40d8276efe569"
)
ANCHOR_GENERATOR_SEQUENCE_SHA256 = (
    "6582894ed4f10b74544bf0197b929b266eb2b63433edc67e8497a640e45451c0"
)
ANCHOR_FINAL_ENERGY = -0.9183809190532167
ANCHOR_ALLOWED_JOB_DELTA_PATHS = (
    "artifact_paths",
    "campaign_id",
    "execution_plan_sha256",
    "package_control_plane_sha256",
    "package_id",
    "sha256",
)
ANCHOR_ALLOWED_RUN_CELL_FUNCTION_DELTAS = (
    "_compiled_resource_projection",
    "_run_g11_bounded_diagnostic",
    "_validate_g6_ra_round",
)

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r100_fresh_package_v1"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r100_fresh_execution_plan_v1"
)
JOB_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r100_fresh_job_v1"
)
SOURCE_AUTHORITY_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r100_source_authority_v1"
)
DELTA_AUDIT_SCHEMA = "source_locked_sensitivity_audit_v1"
EXECUTION_AUTHORIZATION_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r100_execution_authorization_v1"
)

REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ROUTE_IDS = ("append_singleton",)
CANDIDATE_REPRESENTATIONS = {
    "append_singleton": "single_pauli_word_v1",
}
EXPECTED_RESOURCES_BY_NPH = {
    3: {
        "basis": "measured_r70_append_singleton_with_headroom_v1",
        "max_runtime_seconds": 259_200,
        "request_cpus": 1,
        "request_disk_mb": 20_480,
        "request_memory_mb": 16_384,
    },
    7: {
        "basis": "measured_r70_append_singleton_with_headroom_v1",
        "max_runtime_seconds": 259_200,
        "request_cpus": 1,
        "request_disk_mb": 40_960,
        "request_memory_mb": 32_768,
    },
}
EXPECTED_EXECUTION_IDS = tuple(
    f"r100_fresh__{regime_id}__nph{nph}__{route_id}"
    for regime_id, nph in REGIME_CUTOFF_PAIRS
    for route_id in ROUTE_IDS
)
EXPECTED_SOURCE_EXECUTION_IDS = tuple(
    f"core__{regime_id}__nph{nph}__{route_id}"
    for regime_id, nph in REGIME_CUTOFF_PAIRS
    for route_id in ROUTE_IDS
)

ALLOWED_PROTOCOL_DELTA_PATHS = (
    "bundle_materialization.cell_id",
    "bundle_materialization.sha256",
    "horizon",
    "request.execution.stop.maximum_controller_rounds",
    "request.observation.checkpoint.path",
    "request.observation.estimator_ledger.path",
    "route_contract.execution_settings.maximum_controller_rounds",
    "route_contract.sha256",
    "sha256",
    "stopping_rule.maximum_controller_rounds",
)

STATIC_CONTROL_FILES = (
    "build_package.py",
    "derived_protocol.py",
    "package_contract.py",
    "run_cell.py",
    "validate_package.py",
)
GENERATED_CONTROL_FILES = (
    ANCHOR_EVIDENCE_NAME,
    "control_plane_receipt.json",
    "execution_plan.json",
    "horizon_delta_audit.json",
    "queue.tsv",
    SOURCE_ARCHIVE_MANIFEST_NAME,
    SOURCE_ARCHIVE_NAME,
    "source_authority.json",
)
GENERATED_DIRECTORIES = ("jobs",)
MUTABLE_RUNTIME_DIRECTORIES = ("worker_outputs", "worker_receipts")
EXPECTED_ARTIFACT_ROLES = (
    "checkpoint",
    "estimator_ledger",
    "execution_manifest",
    "result",
    "summary",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PackageContractError(ValueError):
    """Raised when any immutable package or source binding drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise PackageContractError("Digest payload already contains sha256.")
    result["sha256"] = canonical_sha256(result)
    return result


def verify_self_digest(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    observed = payload.get("sha256")
    if not isinstance(observed, str) or not _HEX64.fullmatch(observed):
        raise PackageContractError(f"{label} lacks a canonical SHA-256.")
    unsigned = dict(payload)
    del unsigned["sha256"]
    if canonical_sha256(unsigned) != observed:
        raise PackageContractError(f"{label} self digest drifted.")


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"{label} is unavailable or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise PackageContractError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    temporary.replace(path)


def safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PackageContractError(f"{label} is not a safe relative path.")
    return path


def _top_level_function_ast_hashes(path: Path) -> dict[str, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PackageContractError(
            f"Cannot parse anchor control-plane source: {path}"
        ) from exc
    return {
        node.name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode("utf-8")
        ).hexdigest()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _non_function_ast_sha256(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PackageContractError(
            f"Cannot parse anchor control-plane source: {path}"
        ) from exc
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            )
        ],
        type_ignores=[],
    )
    return hashlib.sha256(
        ast.dump(module, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _normalized_anchor_job(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key not in ANCHOR_ALLOWED_JOB_DELTA_PATHS
    }


def _attempt_member_name(
    package_directory_name: str,
    relative: str,
) -> str:
    if relative == "job":
        return (
            "chtc/paper_i_ra_adapt_repair_20260727/"
            f"{package_directory_name}/jobs/{ANCHOR_EXECUTION_ID}.json"
        )
    return f"worker_outputs/{relative}"


def _scan_anchor_attempt(
    *,
    archive_path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
    package_directory_name: str,
    package_id: str,
    job_file_sha256: str,
    job_size_bytes: int,
    job_canonical_sha256: str,
    execution_manifest_sha256: str,
    execution_manifest_size_bytes: int,
    expected_worker_exit_status: int,
    worker_exit_sha256: str,
    expect_worker_receipt: bool,
) -> dict[str, Any]:
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size != archive_size_bytes
        or sha256_file(archive_path) != archive_sha256
    ):
        raise PackageContractError(
            f"Independent-anchor attempt archive drifted: {archive_path}"
        )
    member_names = {
        role: _attempt_member_name(package_directory_name, role)
        for role in (
            "checkpoint.json",
            "execution_manifest.json",
            "job",
            "result.json",
            "summary.json",
            "worker_exit_status.txt",
            "worker_receipt.json",
        )
    }
    role_by_member = {
        member: role for role, member in member_names.items()
    }
    bindings: dict[str, dict[str, Any]] = {}
    retained: dict[str, bytes] = {}
    retain_roles = {
        "checkpoint.json",
        "execution_manifest.json",
        "job",
        "summary.json",
        "worker_exit_status.txt",
        "worker_receipt.json",
    }
    with tarfile.open(archive_path, "r:gz") as bundle:
        for member in bundle:
            role = role_by_member.get(member.name)
            if role is None:
                continue
            if (
                role in bindings
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe or duplicate anchor member: {member.name}"
                )
            stream = bundle.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Cannot read anchor member: {member.name}"
                )
            digest = hashlib.sha256()
            size = 0
            collected = bytearray()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
                if role in retain_roles:
                    collected.extend(block)
            bindings[role] = {
                "path": member.name,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
            if role in retain_roles:
                retained[role] = bytes(collected)
    required_roles = set(member_names) - {"worker_receipt.json"}
    if expect_worker_receipt:
        required_roles.add("worker_receipt.json")
    if not expect_worker_receipt and "worker_receipt.json" in bindings:
        raise PackageContractError(
            f"Unexpected anchor worker receipt: {archive_path}"
        )
    if not required_roles.issubset(bindings):
        missing = sorted(required_roles - set(bindings))
        raise PackageContractError(
            f"Anchor archive lacks required members: {missing}"
        )
    for role, sha256, size_bytes in (
        ("result.json", ANCHOR_RESULT_SHA256, ANCHOR_RESULT_SIZE_BYTES),
        ("summary.json", ANCHOR_SUMMARY_SHA256, ANCHOR_SUMMARY_SIZE_BYTES),
        (
            "checkpoint.json",
            ANCHOR_CHECKPOINT_SHA256,
            ANCHOR_CHECKPOINT_SIZE_BYTES,
        ),
        ("execution_manifest.json", execution_manifest_sha256,
         execution_manifest_size_bytes),
        ("job", job_file_sha256, job_size_bytes),
        ("worker_exit_status.txt", worker_exit_sha256, 2),
    ):
        observed = bindings[role]
        if (
            observed["sha256"] != sha256
            or int(observed["size_bytes"]) != int(size_bytes)
        ):
            raise PackageContractError(
                f"Anchor member binding drifted: {archive_path}#{role}"
            )
    try:
        manifest = json.loads(
            retained["execution_manifest.json"].decode("utf-8")
        )
        job = json.loads(retained["job"].decode("utf-8"))
        summary = json.loads(retained["summary.json"].decode("utf-8"))
        checkpoint = json.loads(
            retained["checkpoint.json"].decode("utf-8")
        )
        worker_receipt = (
            json.loads(retained["worker_receipt.json"].decode("utf-8"))
            if expect_worker_receipt
            else None
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(
            f"Anchor JSON member is invalid: {archive_path}"
        ) from exc
    if not all(
        isinstance(payload, dict)
        for payload in (manifest, job, summary, checkpoint)
    ):
        raise PackageContractError("Anchor JSON member is not an object.")
    if expect_worker_receipt and not isinstance(worker_receipt, dict):
        raise PackageContractError("Source worker receipt is not an object.")
    verify_self_digest(manifest, label="anchor execution manifest")
    verify_self_digest(job, label="anchor job")
    if worker_receipt is not None:
        verify_self_digest(worker_receipt, label="source worker receipt")
    if (
        retained["worker_exit_status.txt"]
        != f"{expected_worker_exit_status}\n".encode("ascii")
        or manifest.get("status") != "passed"
        or manifest.get("package_id") != package_id
        or manifest.get("execution_id") != ANCHOR_EXECUTION_ID
        or manifest.get("execution_entrypoint") != "run_append_adapt"
        or manifest.get("protocol_sha256") != ANCHOR_PROTOCOL_SHA256
        or manifest.get("maximum_controller_rounds_override") is not None
        or manifest.get("paper_facing_result_allowed") is not True
        or manifest.get("job_spec_sha256") != job_canonical_sha256
        or manifest.get("g11_bounded_replay_diagnostic", {}).get(
            "selected"
        )
        is not False
        or job.get("sha256") != job_canonical_sha256
        or job.get("execution_id") != ANCHOR_EXECUTION_ID
        or job.get("execution_entrypoint") != "run_append_adapt"
        or job.get("protocol", {}).get("canonical_sha256")
        != ANCHOR_PROTOCOL_SHA256
        or job.get("source_archive_sha256")
        != ANCHOR_SOURCE_ARCHIVE_SHA256
        or summary.get("protocol_sha256") != ANCHOR_PROTOCOL_SHA256
        or summary.get("protocol_horizon") != SOURCE_HORIZON
        or summary.get("controller_rounds_completed") != SOURCE_HORIZON
        or summary.get("stop_reason") != "maximum_controller_rounds"
        or summary.get("candidate_representation")
        != "single_pauli_word_v1"
        or summary.get("algorithm_id") != "paper_i_append_adapt_v1"
        or len(summary.get("accepted_operator_labels", []))
        != SOURCE_HORIZON
        or len(summary.get("accepted_generator_identities", []))
        != SOURCE_HORIZON
    ):
        raise PackageContractError(
            f"Anchor scientific semantics drifted: {archive_path}"
        )
    if expect_worker_receipt and (
        bindings["worker_receipt.json"]["sha256"]
        != SOURCE_WORKER_RECEIPT_SHA256
        or bindings["worker_receipt.json"]["size_bytes"]
        != SOURCE_WORKER_RECEIPT_SIZE_BYTES
        or worker_receipt.get("status") != "passed"
        or worker_receipt.get("execution_id") != ANCHOR_EXECUTION_ID
        or worker_receipt.get("scheduler_cluster_id") != SOURCE_CLUSTER_ID
        or worker_receipt.get("scheduler_proc_id") != SOURCE_PROC_ID
    ):
        raise PackageContractError("Source worker receipt binding drifted.")
    outputs = manifest.get("output_payloads")
    if not isinstance(outputs, Mapping):
        raise PackageContractError("Anchor manifest has no output bindings.")
    for role in ("checkpoint", "result", "summary"):
        member_role = f"{role}.json"
        if outputs.get(role) != {
            "sha256": bindings[member_role]["sha256"],
            "size_bytes": bindings[member_role]["size_bytes"],
        }:
            raise PackageContractError(
                f"Anchor manifest/member mismatch: {archive_path}#{role}"
            )
    return {
        "archive": {
            "path": archive_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": archive_sha256,
            "size_bytes": archive_size_bytes,
        },
        "checkpoint": checkpoint,
        "execution_manifest": manifest,
        "job": job,
        "members": bindings,
        "summary": summary,
        "worker_exit_status": expected_worker_exit_status,
        "worker_receipt": worker_receipt,
    }


def materialize_anchor_evidence() -> dict[str, Any]:
    """Recompute the completed independent r50 anchor/source comparison."""

    anchor_package_dir = (
        REPO_ROOT
        / "chtc/paper_i_ra_adapt_repair_20260727"
        / ANCHOR_PACKAGE_DIRECTORY_NAME
    )
    source_package_dir = SOURCE_PACKAGE_DIR
    anchor = _scan_anchor_attempt(
        archive_path=REPO_ROOT / ANCHOR_ATTEMPT_RELATIVE_PATH,
        archive_sha256=ANCHOR_ATTEMPT_SHA256,
        archive_size_bytes=ANCHOR_ATTEMPT_SIZE_BYTES,
        package_directory_name=ANCHOR_PACKAGE_DIRECTORY_NAME,
        package_id=ANCHOR_PACKAGE_ID,
        job_file_sha256=ANCHOR_JOB_FILE_SHA256,
        job_size_bytes=ANCHOR_JOB_SIZE_BYTES,
        job_canonical_sha256=ANCHOR_JOB_CANONICAL_SHA256,
        execution_manifest_sha256=ANCHOR_EXECUTION_MANIFEST_SHA256,
        execution_manifest_size_bytes=(
            ANCHOR_EXECUTION_MANIFEST_SIZE_BYTES
        ),
        expected_worker_exit_status=ANCHOR_WORKER_EXIT_STATUS,
        worker_exit_sha256=ANCHOR_WORKER_EXIT_SHA256,
        expect_worker_receipt=False,
    )
    source = _scan_anchor_attempt(
        archive_path=REPO_ROOT / SOURCE_ATTEMPT_RELATIVE_PATH,
        archive_sha256=SOURCE_ATTEMPT_SHA256,
        archive_size_bytes=SOURCE_ATTEMPT_SIZE_BYTES,
        package_directory_name=SOURCE_PACKAGE_DIRECTORY_NAME,
        package_id=SOURCE_PACKAGE_ID,
        job_file_sha256=SOURCE_ANCHOR_JOB_FILE_SHA256,
        job_size_bytes=SOURCE_ANCHOR_JOB_SIZE_BYTES,
        job_canonical_sha256=SOURCE_ANCHOR_JOB_CANONICAL_SHA256,
        execution_manifest_sha256=SOURCE_EXECUTION_MANIFEST_SHA256,
        execution_manifest_size_bytes=(
            SOURCE_EXECUTION_MANIFEST_SIZE_BYTES
        ),
        expected_worker_exit_status=SOURCE_WORKER_EXIT_STATUS,
        worker_exit_sha256=SOURCE_WORKER_EXIT_SHA256,
        expect_worker_receipt=True,
    )
    for package_dir, expected_run_cell_sha256 in (
        (anchor_package_dir, ANCHOR_RUN_CELL_SHA256),
        (source_package_dir, SOURCE_RUN_CELL_SHA256),
    ):
        if (
            sha256_file(package_dir / "run_cell.py")
            != expected_run_cell_sha256
            or sha256_file(package_dir / "source_locked.tar.gz")
            != ANCHOR_SOURCE_ARCHIVE_SHA256
            or sha256_file(package_dir / "execute_source_locked_job.sh")
            != ANCHOR_EXECUTE_WRAPPER_SHA256
            or ANCHOR_IMAGE_SHA256
            not in (package_dir / "submit.sub").read_text(encoding="utf-8")
        ):
            raise PackageContractError(
                f"Anchor executable-state binding drifted: {package_dir}"
            )
    anchor_functions = _top_level_function_ast_hashes(
        anchor_package_dir / "run_cell.py"
    )
    source_functions = _top_level_function_ast_hashes(
        source_package_dir / "run_cell.py"
    )
    changed_functions = sorted(
        name
        for name in set(anchor_functions) | set(source_functions)
        if anchor_functions.get(name) != source_functions.get(name)
    )
    if (
        changed_functions
        != sorted(ANCHOR_ALLOWED_RUN_CELL_FUNCTION_DELTAS)
        or anchor_functions.get("_invoke")
        != source_functions.get("_invoke")
        or source_functions.get("_invoke")
        != ANCHOR_PRIMARY_INVOKE_AST_SHA256
    ):
        raise PackageContractError(
            "Anchor run_cell scientific/control-plane separation drifted."
        )
    anchor_non_function_ast_sha256 = _non_function_ast_sha256(
        anchor_package_dir / "run_cell.py"
    )
    source_non_function_ast_sha256 = _non_function_ast_sha256(
        source_package_dir / "run_cell.py"
    )
    if (
        anchor_non_function_ast_sha256
        != source_non_function_ast_sha256
        or source_non_function_ast_sha256
        != ANCHOR_NON_FUNCTION_AST_SHA256
    ):
        raise PackageContractError(
            "Anchor run_cell module-level semantics drifted."
        )
    anchor_normalized_job = _normalized_anchor_job(anchor["job"])
    source_normalized_job = _normalized_anchor_job(source["job"])
    anchor_summary = anchor["summary"]
    source_summary = source["summary"]
    if (
        anchor_normalized_job != source_normalized_job
        or anchor["members"]["result.json"]["sha256"]
        != source["members"]["result.json"]["sha256"]
        or anchor["members"]["result.json"]["size_bytes"]
        != source["members"]["result.json"]["size_bytes"]
        or anchor["members"]["summary.json"]["sha256"]
        != source["members"]["summary.json"]["sha256"]
        or anchor["members"]["checkpoint.json"]["sha256"]
        != source["members"]["checkpoint.json"]["sha256"]
        or anchor_summary != source_summary
        or anchor["checkpoint"] != source["checkpoint"]
    ):
        raise PackageContractError(
            "Independent round-50 anchor did not reproduce the v6 source."
        )
    normalized_scientific_job_sha256 = canonical_sha256(
        source_normalized_job
    )
    operator_sequence_sha256 = canonical_sha256(
        source_summary["accepted_operator_labels"]
    )
    generator_sequence_sha256 = canonical_sha256(
        source_summary["accepted_generator_identities"]
    )
    if (
        normalized_scientific_job_sha256
        != ANCHOR_NORMALIZED_SCIENTIFIC_JOB_SHA256
        or operator_sequence_sha256 != ANCHOR_OPERATOR_SEQUENCE_SHA256
        or generator_sequence_sha256
        != ANCHOR_GENERATOR_SEQUENCE_SHA256
        or anchor_summary["final_energy"] != ANCHOR_FINAL_ENERGY
        or source_summary["final_energy"] != ANCHOR_FINAL_ENERGY
    ):
        raise PackageContractError(
            "Independent anchor normalized scientific evidence drifted."
        )
    return digested(
        {
            "schema": ANCHOR_EVIDENCE_SCHEMA,
            "status": "passed",
            "anchor_role": (
                "preexisting_independent_completed_scientific_payload_"
                "with_post_invoke_worker_failure_v1"
            ),
            "source_value": SOURCE_HORIZON,
            "execution_id": ANCHOR_EXECUTION_ID,
            "protocol_sha256": ANCHOR_PROTOCOL_SHA256,
            "locked_source_archive_sha256": (
                ANCHOR_SOURCE_ARCHIVE_SHA256
            ),
            "execution_image_sha256": ANCHOR_IMAGE_SHA256,
            "anchor_execution": {
                "package_id": ANCHOR_PACKAGE_ID,
                "cluster_id": ANCHOR_CLUSTER_ID,
                "proc_id": ANCHOR_PROC_ID,
                "completed_utc": anchor["execution_manifest"][
                    "completed_utc"
                ],
                "archive": anchor["archive"],
                "job": anchor["members"]["job"],
                "job_canonical_sha256": ANCHOR_JOB_CANONICAL_SHA256,
                "execution_manifest": anchor["members"][
                    "execution_manifest.json"
                ],
                "scientific_execution_manifest_status": (
                    anchor["execution_manifest"]["status"]
                ),
                "worker_exit_status": anchor["worker_exit_status"],
                "worker_receipt": None,
                "full_worker_attempt_passed": False,
                "result": anchor["members"]["result.json"],
                "summary": anchor["members"]["summary.json"],
                "checkpoint": anchor["members"]["checkpoint.json"],
            },
            "source_execution": {
                "package_id": SOURCE_PACKAGE_ID,
                "cluster_id": SOURCE_CLUSTER_ID,
                "proc_id": SOURCE_PROC_ID,
                "completed_utc": source["execution_manifest"][
                    "completed_utc"
                ],
                "archive": source["archive"],
                "job": source["members"]["job"],
                "job_canonical_sha256": (
                    SOURCE_ANCHOR_JOB_CANONICAL_SHA256
                ),
                "execution_manifest": source["members"][
                    "execution_manifest.json"
                ],
                "scientific_execution_manifest_status": (
                    source["execution_manifest"]["status"]
                ),
                "worker_exit_status": source["worker_exit_status"],
                "worker_receipt": source["members"][
                    "worker_receipt.json"
                ],
                "full_worker_attempt_passed": True,
                "result": source["members"]["result.json"],
                "summary": source["members"]["summary.json"],
                "checkpoint": source["members"]["checkpoint.json"],
            },
            "comparison": {
                "independent_execution": True,
                "different_package_id": True,
                "different_cluster_id": True,
                "normalized_scientific_job_match": True,
                "normalized_scientific_job_sha256": (
                    normalized_scientific_job_sha256
                ),
                "normalized_ignored_control_fields": list(
                    ANCHOR_ALLOWED_JOB_DELTA_PATHS
                ),
                "protocol_sha256_match": True,
                "locked_source_archive_sha256_match": True,
                "execution_image_sha256_match": True,
                "execute_wrapper_sha256_match": True,
                "primary_invoke_ast_sha256_match": True,
                "primary_invoke_ast_sha256": source_functions["_invoke"],
                "module_non_function_ast_sha256_match": True,
                "module_non_function_ast_sha256": (
                    source_non_function_ast_sha256
                ),
                "run_cell_control_plane_delta": {
                    "anchor_run_cell_sha256": ANCHOR_RUN_CELL_SHA256,
                    "source_run_cell_sha256": SOURCE_RUN_CELL_SHA256,
                    "changed_top_level_functions": changed_functions,
                    "allowed_changed_top_level_functions": list(
                        ANCHOR_ALLOWED_RUN_CELL_FUNCTION_DELTAS
                    ),
                    "classification": (
                        "post_result_reporting_ra_only_validation_and_"
                        "unselected_diagnostic_validation_v1"
                    ),
                    "anchor_worker_exit_status": (
                        ANCHOR_WORKER_EXIT_STATUS
                    ),
                    "source_worker_exit_status": (
                        SOURCE_WORKER_EXIT_STATUS
                    ),
                    "anchor_scientific_payload_complete": True,
                    "anchor_worker_receipt_present": False,
                    "source_worker_receipt_present": True,
                    "failure_stage_bound": (
                        "after_primary_scientific_payload_before_"
                        "worker_receipt"
                    ),
                    "failure_cause_asserted": False,
                    "failure_cause_note": (
                        "The archived attempt contains no stderr member; "
                        "the exact exception is therefore not asserted."
                    ),
                    "anchor_route_diagnostic_selected": False,
                    "scientific_primary_invoke_changed": False,
                    "scientific_protocol_or_result_changed": False,
                },
                "result_bytes_match": True,
                "scientific_payload_reproduces_source": True,
                "full_worker_attempt_reproduces_source": False,
                "summary_bytes_match": True,
                "checkpoint_bytes_match": True,
                "operator_sequence_match": True,
                "operator_sequence_sha256": operator_sequence_sha256,
                "generator_sequence_match": True,
                "generator_sequence_sha256": generator_sequence_sha256,
                "metric_match": True,
                "metric_name": "final_energy",
                "anchor_metric": anchor_summary["final_energy"],
                "source_metric": source_summary["final_energy"],
                "metric_abs_diff": 0.0,
                "stopping_condition_match": True,
                "stop_reason": "maximum_controller_rounds",
                "controller_rounds_completed": SOURCE_HORIZON,
                "non_swept_settings_diff": [],
            },
            "new_scientific_execution_performed": False,
            "completed_evidence_reused": True,
        }
    )


def validate_anchor_evidence(
    evidence: Mapping[str, Any],
    *,
    full_attempt_scan: bool,
) -> None:
    verify_self_digest(evidence, label="independent anchor evidence")
    anchor = evidence.get("anchor_execution")
    source = evidence.get("source_execution")
    comparison = evidence.get("comparison")
    if not all(
        isinstance(value, Mapping)
        for value in (anchor, source, comparison)
    ):
        raise PackageContractError("Independent anchor evidence is incomplete.")
    source_worker_receipt = source.get("worker_receipt")
    anchor_result = anchor.get("result")
    source_result = source.get("result")
    anchor_summary = anchor.get("summary")
    source_summary = source.get("summary")
    anchor_checkpoint = anchor.get("checkpoint")
    source_checkpoint = source.get("checkpoint")
    anchor_manifest = anchor.get("execution_manifest")
    source_manifest = source.get("execution_manifest")
    anchor_job = anchor.get("job")
    source_job = source.get("job")
    control_delta = comparison.get("run_cell_control_plane_delta")
    if not all(
        isinstance(value, Mapping)
        for value in (
            source_worker_receipt,
            anchor_result,
            source_result,
            anchor_summary,
            source_summary,
            anchor_checkpoint,
            source_checkpoint,
            anchor_manifest,
            source_manifest,
            anchor_job,
            source_job,
            control_delta,
        )
    ):
        raise PackageContractError("Independent anchor evidence is incomplete.")
    if (
        evidence.get("schema") != ANCHOR_EVIDENCE_SCHEMA
        or evidence.get("status") != "passed"
        or evidence.get("anchor_role")
        != (
            "preexisting_independent_completed_scientific_payload_"
            "with_post_invoke_worker_failure_v1"
        )
        or evidence.get("source_value") != SOURCE_HORIZON
        or evidence.get("execution_id") != ANCHOR_EXECUTION_ID
        or evidence.get("protocol_sha256") != ANCHOR_PROTOCOL_SHA256
        or evidence.get("locked_source_archive_sha256")
        != ANCHOR_SOURCE_ARCHIVE_SHA256
        or evidence.get("execution_image_sha256") != ANCHOR_IMAGE_SHA256
        or evidence.get("new_scientific_execution_performed") is not False
        or evidence.get("completed_evidence_reused") is not True
        or anchor.get("package_id") != ANCHOR_PACKAGE_ID
        or anchor.get("cluster_id") != ANCHOR_CLUSTER_ID
        or anchor.get("proc_id") != ANCHOR_PROC_ID
        or anchor.get("completed_utc") != ANCHOR_COMPLETED_UTC
        or source.get("package_id") != SOURCE_PACKAGE_ID
        or source.get("cluster_id") != SOURCE_CLUSTER_ID
        or source.get("proc_id") != SOURCE_PROC_ID
        or source.get("completed_utc") != SOURCE_COMPLETED_UTC
        or anchor.get("archive")
        != {
            "path": ANCHOR_ATTEMPT_RELATIVE_PATH,
            "sha256": ANCHOR_ATTEMPT_SHA256,
            "size_bytes": ANCHOR_ATTEMPT_SIZE_BYTES,
        }
        or source.get("archive")
        != {
            "path": SOURCE_ATTEMPT_RELATIVE_PATH,
            "sha256": SOURCE_ATTEMPT_SHA256,
            "size_bytes": SOURCE_ATTEMPT_SIZE_BYTES,
        }
        or anchor.get("scientific_execution_manifest_status") != "passed"
        or source.get("scientific_execution_manifest_status") != "passed"
        or anchor.get("worker_exit_status") != ANCHOR_WORKER_EXIT_STATUS
        or source.get("worker_exit_status") != SOURCE_WORKER_EXIT_STATUS
        or anchor.get("worker_receipt") is not None
        or source_worker_receipt
        != {
            "path": "worker_outputs/worker_receipt.json",
            "sha256": SOURCE_WORKER_RECEIPT_SHA256,
            "size_bytes": SOURCE_WORKER_RECEIPT_SIZE_BYTES,
        }
        or anchor.get("full_worker_attempt_passed") is not False
        or source.get("full_worker_attempt_passed") is not True
        or anchor_manifest
        != {
            "path": "worker_outputs/execution_manifest.json",
            "sha256": ANCHOR_EXECUTION_MANIFEST_SHA256,
            "size_bytes": ANCHOR_EXECUTION_MANIFEST_SIZE_BYTES,
        }
        or source_manifest
        != {
            "path": "worker_outputs/execution_manifest.json",
            "sha256": SOURCE_EXECUTION_MANIFEST_SHA256,
            "size_bytes": SOURCE_EXECUTION_MANIFEST_SIZE_BYTES,
        }
        or anchor_job
        != {
            "path": _attempt_member_name(
                ANCHOR_PACKAGE_DIRECTORY_NAME, "job"
            ),
            "sha256": ANCHOR_JOB_FILE_SHA256,
            "size_bytes": ANCHOR_JOB_SIZE_BYTES,
        }
        or source_job
        != {
            "path": _attempt_member_name(
                SOURCE_PACKAGE_DIRECTORY_NAME, "job"
            ),
            "sha256": SOURCE_ANCHOR_JOB_FILE_SHA256,
            "size_bytes": SOURCE_ANCHOR_JOB_SIZE_BYTES,
        }
        or anchor.get("job_canonical_sha256")
        != ANCHOR_JOB_CANONICAL_SHA256
        or source.get("job_canonical_sha256")
        != SOURCE_ANCHOR_JOB_CANONICAL_SHA256
        or anchor_result
        != {
            "path": "worker_outputs/result.json",
            "sha256": ANCHOR_RESULT_SHA256,
            "size_bytes": ANCHOR_RESULT_SIZE_BYTES,
        }
        or source_result != anchor_result
        or anchor_summary
        != {
            "path": "worker_outputs/summary.json",
            "sha256": ANCHOR_SUMMARY_SHA256,
            "size_bytes": ANCHOR_SUMMARY_SIZE_BYTES,
        }
        or source_summary != anchor_summary
        or anchor_checkpoint
        != {
            "path": "worker_outputs/checkpoint.json",
            "sha256": ANCHOR_CHECKPOINT_SHA256,
            "size_bytes": ANCHOR_CHECKPOINT_SIZE_BYTES,
        }
        or source_checkpoint != anchor_checkpoint
        or comparison.get("independent_execution") is not True
        or comparison.get("different_package_id") is not True
        or comparison.get("different_cluster_id") is not True
        or comparison.get("normalized_scientific_job_match") is not True
        or comparison.get("normalized_scientific_job_sha256")
        != ANCHOR_NORMALIZED_SCIENTIFIC_JOB_SHA256
        or comparison.get("normalized_ignored_control_fields")
        != list(ANCHOR_ALLOWED_JOB_DELTA_PATHS)
        or comparison.get("protocol_sha256_match") is not True
        or comparison.get("locked_source_archive_sha256_match")
        is not True
        or comparison.get("execution_image_sha256_match") is not True
        or comparison.get("execute_wrapper_sha256_match") is not True
        or comparison.get("primary_invoke_ast_sha256_match") is not True
        or comparison.get("primary_invoke_ast_sha256")
        != ANCHOR_PRIMARY_INVOKE_AST_SHA256
        or comparison.get("module_non_function_ast_sha256_match")
        is not True
        or comparison.get("module_non_function_ast_sha256")
        != ANCHOR_NON_FUNCTION_AST_SHA256
        or comparison.get("scientific_payload_reproduces_source")
        is not True
        or comparison.get("full_worker_attempt_reproduces_source")
        is not False
        or comparison.get("result_bytes_match") is not True
        or comparison.get("summary_bytes_match") is not True
        or comparison.get("checkpoint_bytes_match") is not True
        or comparison.get("operator_sequence_match") is not True
        or comparison.get("operator_sequence_sha256")
        != ANCHOR_OPERATOR_SEQUENCE_SHA256
        or comparison.get("generator_sequence_match") is not True
        or comparison.get("generator_sequence_sha256")
        != ANCHOR_GENERATOR_SEQUENCE_SHA256
        or comparison.get("metric_match") is not True
        or comparison.get("metric_name") != "final_energy"
        or comparison.get("anchor_metric") != ANCHOR_FINAL_ENERGY
        or comparison.get("source_metric") != ANCHOR_FINAL_ENERGY
        or comparison.get("metric_abs_diff") != 0.0
        or comparison.get("stopping_condition_match") is not True
        or comparison.get("stop_reason") != "maximum_controller_rounds"
        or comparison.get("controller_rounds_completed") != SOURCE_HORIZON
        or comparison.get("non_swept_settings_diff") != []
        or control_delta.get("anchor_run_cell_sha256")
        != ANCHOR_RUN_CELL_SHA256
        or control_delta.get("source_run_cell_sha256")
        != SOURCE_RUN_CELL_SHA256
        or control_delta.get("changed_top_level_functions")
        != sorted(ANCHOR_ALLOWED_RUN_CELL_FUNCTION_DELTAS)
        or control_delta.get("allowed_changed_top_level_functions")
        != list(ANCHOR_ALLOWED_RUN_CELL_FUNCTION_DELTAS)
        or control_delta.get("classification")
        != (
            "post_result_reporting_ra_only_validation_and_"
            "unselected_diagnostic_validation_v1"
        )
        or control_delta.get("scientific_primary_invoke_changed") is not False
        or control_delta.get("anchor_worker_exit_status")
        != ANCHOR_WORKER_EXIT_STATUS
        or control_delta.get("source_worker_exit_status")
        != SOURCE_WORKER_EXIT_STATUS
        or control_delta.get("anchor_scientific_payload_complete")
        is not True
        or control_delta.get("anchor_worker_receipt_present")
        is not False
        or control_delta.get("source_worker_receipt_present")
        is not True
        or control_delta.get("failure_stage_bound")
        != "after_primary_scientific_payload_before_worker_receipt"
        or control_delta.get("failure_cause_asserted") is not False
        or control_delta.get("failure_cause_note")
        != (
            "The archived attempt contains no stderr member; "
            "the exact exception is therefore not asserted."
        )
        or control_delta.get("anchor_route_diagnostic_selected") is not False
        or control_delta.get("scientific_protocol_or_result_changed")
        is not False
    ):
        raise PackageContractError("Independent anchor evidence drifted.")
    if full_attempt_scan and dict(evidence) != materialize_anchor_evidence():
        raise PackageContractError(
            "Independent anchor archive/member recomputation drifted."
        )


def audit_anchor_payload(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    anchor = evidence["anchor_execution"]
    source = evidence["source_execution"]
    comparison = evidence["comparison"]
    return {
        "kind": (
            "preexisting_independent_completed_scientific_payload_"
            "anchor_v1"
        ),
        "value": SOURCE_HORIZON,
        "anchor_evidence_path": ANCHOR_EVIDENCE_NAME,
        "anchor_evidence_sha256": evidence["sha256"],
        "anchor_result_json": (
            f"{anchor['archive']['path']}#{anchor['result']['path']}"
        ),
        "anchor_result_sha256": anchor["result"]["sha256"],
        "source_result_json": (
            f"{source['archive']['path']}#{source['result']['path']}"
        ),
        "source_result_sha256": source["result"]["sha256"],
        "anchor_summary_sha256": anchor["summary"]["sha256"],
        "source_summary_sha256": source["summary"]["sha256"],
        "anchor_checkpoint_sha256": anchor["checkpoint"]["sha256"],
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "protocol_sha256": ANCHOR_PROTOCOL_SHA256,
        "anchor_reproduces_source": True,
        "reproduction_scope": "scientific_payload_only",
        "anchor_scientific_execution_manifest_status": (
            anchor["scientific_execution_manifest_status"]
        ),
        "source_scientific_execution_manifest_status": (
            source["scientific_execution_manifest_status"]
        ),
        "anchor_worker_exit_status": anchor["worker_exit_status"],
        "source_worker_exit_status": source["worker_exit_status"],
        "anchor_full_worker_attempt_passed": False,
        "source_full_worker_attempt_passed": True,
        "anchor_worker_receipt_present": False,
        "source_worker_receipt_sha256": source["worker_receipt"]["sha256"],
        "independent_execution": True,
        "result_bytes_match": True,
        "summary_bytes_match": True,
        "checkpoint_bytes_match": True,
        "operator_sequence_match": True,
        "operator_sequence_sha256": comparison[
            "operator_sequence_sha256"
        ],
        "generator_sequence_match": True,
        "generator_sequence_sha256": comparison[
            "generator_sequence_sha256"
        ],
        "metric_name": comparison["metric_name"],
        "anchor_metric": comparison["anchor_metric"],
        "source_metric": comparison["source_metric"],
        "metric_abs_diff": comparison["metric_abs_diff"],
        "stopping_condition_match": True,
        "stop_reason": comparison["stop_reason"],
        "controller_rounds_completed": SOURCE_HORIZON,
        "normalized_scientific_job_match": True,
        "run_cell_control_plane_delta": comparison[
            "run_cell_control_plane_delta"
        ],
        "non_swept_settings_diff": [],
        "source_package_validation_status": "passed",
        "new_scientific_execution_performed": False,
        "completed_evidence_reused": True,
    }


def _archive_member_map(
    source_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    rows = source_manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise PackageContractError("Source archive manifest has no members.")
    by_path: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PackageContractError("Malformed source archive member.")
        relative = str(row.get("path", ""))
        safe_relative_path(relative, label="source archive member")
        if relative in by_path:
            raise PackageContractError(
                f"Duplicate source archive member: {relative}"
            )
        if (
            not isinstance(row.get("sha256"), str)
            or not _HEX64.fullmatch(str(row["sha256"]))
            or int(row.get("size_bytes", -1)) < 0
        ):
            raise PackageContractError(
                f"Malformed source archive binding: {relative}"
            )
        by_path[relative] = row
    if int(source_manifest.get("member_count", -1)) != len(by_path):
        raise PackageContractError("Source archive member count drifted.")
    return by_path


def validate_source_archive(
    *,
    package_dir: Path = PACKAGE_DIR,
    full_member_scan: bool,
) -> dict[str, Any]:
    archive = package_dir / SOURCE_ARCHIVE_NAME
    manifest_path = package_dir / SOURCE_ARCHIVE_MANIFEST_NAME
    manifest = load_json_object(
        manifest_path, label="source archive manifest"
    )
    verify_self_digest(manifest, label="source archive manifest")
    if (
        manifest.get("sha256") != SOURCE_ARCHIVE_MANIFEST_SHA256
        or sha256_file(manifest_path)
        != SOURCE_ARCHIVE_MANIFEST_FILE_SHA256
        or manifest.get("archive")
        != {
            "path": "source_locked.tar.gz",
            "sha256": SOURCE_ARCHIVE_SHA256,
            "size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
        }
        or sha256_file(archive) != SOURCE_ARCHIVE_SHA256
        or archive.stat().st_size != SOURCE_ARCHIVE_SIZE_BYTES
    ):
        raise PackageContractError("Source archive authority drifted.")
    expected = _archive_member_map(manifest)
    if full_member_scan:
        observed: dict[str, dict[str, Any]] = {}
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                relative = member.name
                safe_relative_path(relative, label="tar member")
                if not member.isfile() or member.issym() or member.islnk():
                    raise PackageContractError(
                        f"Source archive member is not regular: {relative}"
                    )
                if relative in observed:
                    raise PackageContractError(
                        f"Duplicate tar member: {relative}"
                    )
                stream = bundle.extractfile(member)
                if stream is None:
                    raise PackageContractError(
                        f"Cannot read tar member: {relative}"
                    )
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                observed[relative] = {
                    "sha256": digest.hexdigest(),
                    "size_bytes": size,
                }
        if set(observed) != set(expected):
            raise PackageContractError(
                "Source archive membership drifted from its manifest."
            )
        for relative, binding in observed.items():
            declared = expected[relative]
            if (
                binding["sha256"] != declared["sha256"]
                or binding["size_bytes"] != int(declared["size_bytes"])
            ):
                raise PackageContractError(
                    f"Source archive member drifted: {relative}"
                )
    return manifest


def _expected_source_for_execution(execution_id: str) -> str:
    if execution_id not in EXPECTED_EXECUTION_IDS:
        raise PackageContractError(f"Unknown execution id: {execution_id}")
    return "core__" + execution_id.removeprefix("r100_fresh__")


def _expected_execution_semantics(
    execution_id: str,
) -> dict[str, Any]:
    for regime_id, nph in REGIME_CUTOFF_PAIRS:
        for route_id in ROUTE_IDS:
            expected = f"r100_fresh__{regime_id}__nph{nph}__{route_id}"
            if execution_id == expected:
                return {
                    "regime_id": regime_id,
                    "nph": nph,
                    "route_id": route_id,
                    "candidate_representation": (
                        CANDIDATE_REPRESENTATIONS[route_id]
                    ),
                    "source_execution_id": (
                        f"core__{regime_id}__nph{nph}__{route_id}"
                    ),
                    "source_lock_id": (
                        f"{regime_id}__nph{nph}__{route_id}"
                    ),
                    "resources": EXPECTED_RESOURCES_BY_NPH[nph],
                }
    raise PackageContractError(f"Unknown execution id: {execution_id}")


def _expected_artifact_paths(execution_id: str) -> dict[str, str]:
    root = (
        "raw_outputs/"
        "paper_i_append_adapt_stationary_singleton6_r100_fresh_20260808_v1/"
        f"{execution_id}"
    )
    return {
        "checkpoint": f"{root}/checkpoint.json",
        "estimator_ledger": f"{root}/estimator_ledger.json",
        "execution_manifest": f"{root}/execution_manifest.json",
        "result": f"{root}/result.json",
        "summary": f"{root}/summary.json",
    }


def _validate_source_authority_row(
    row: Mapping[str, Any],
    *,
    execution_id: str,
    source_members: Mapping[str, Mapping[str, Any]],
) -> None:
    semantics = _expected_execution_semantics(execution_id)
    source_job = row.get("source_job")
    source_protocol = row.get("source_protocol")
    if not isinstance(source_job, Mapping) or not isinstance(
        source_protocol, Mapping
    ):
        raise PackageContractError(
            f"Source-authority row is incomplete: {execution_id}"
        )
    protocol_path = str(source_protocol.get("path", ""))
    archive_binding = source_members.get(protocol_path)
    if (
        set(row)
        != {
            "execution_id",
            "source_execution_id",
            "source_job",
            "source_lock_id",
            "source_protocol",
        }
        or row.get("execution_id") != execution_id
        or row.get("source_execution_id")
        != semantics["source_execution_id"]
        or row.get("source_lock_id") != semantics["source_lock_id"]
        or set(source_job)
        != {"canonical_sha256", "path", "sha256", "size_bytes"}
        or source_job.get("path")
        != f"jobs/{semantics['source_execution_id']}.json"
        or not isinstance(source_job.get("canonical_sha256"), str)
        or not _HEX64.fullmatch(str(source_job["canonical_sha256"]))
        or not isinstance(source_job.get("sha256"), str)
        or not _HEX64.fullmatch(str(source_job["sha256"]))
        or int(source_job.get("size_bytes", -1)) <= 0
        or set(source_protocol)
        != {"canonical_sha256", "path", "sha256", "size_bytes"}
        or archive_binding is None
        or source_protocol.get("sha256") != archive_binding.get("sha256")
        or int(source_protocol.get("size_bytes", -1))
        != int(archive_binding.get("size_bytes", -2))
        or not isinstance(source_protocol.get("canonical_sha256"), str)
        or not _HEX64.fullmatch(str(source_protocol["canonical_sha256"]))
    ):
        raise PackageContractError(
            f"Source-authority row drifted: {execution_id}"
        )


def _validate_delta_row(
    row: Mapping[str, Any],
    *,
    execution_id: str,
    source_authority_row: Mapping[str, Any],
) -> None:
    semantics = _expected_execution_semantics(execution_id)
    changed_values = row.get("changed_values")
    if not isinstance(changed_values, list):
        raise PackageContractError(
            f"Delta row lacks changed values: {execution_id}"
        )
    changed_by_path = {
        str(change.get("path")): change
        for change in changed_values
        if isinstance(change, Mapping)
    }
    source_protocol = source_authority_row["source_protocol"]
    if (
        row.get("execution_id") != execution_id
        or row.get("source_execution_id")
        != semantics["source_execution_id"]
        or row.get("regime_id") != semantics["regime_id"]
        or int(row.get("nph", -1)) != semantics["nph"]
        or row.get("route_id") != semantics["route_id"]
        or row.get("candidate_representation")
        != semantics["candidate_representation"]
        or row.get("source_horizon") != SOURCE_HORIZON
        or row.get("target_horizon") != TARGET_HORIZON
        or row.get("fresh_start") is not True
        or row.get("resume_claimed") is not False
        or row.get("source_checkpoint_consumed") is not False
        or row.get("source_result_consumed") is not False
        or row.get("source_baseline_consumption_status") != "passed"
        or row.get("source_protocol_path") != source_protocol["path"]
        or row.get("source_protocol_sha256")
        != source_protocol["canonical_sha256"]
        or not isinstance(row.get("derived_protocol_sha256"), str)
        or not _HEX64.fullmatch(str(row["derived_protocol_sha256"]))
        or row.get("normalized_non_horizon_settings_match") is not True
        or not isinstance(
            row.get("normalized_non_horizon_settings_sha256"), str
        )
        or not _HEX64.fullmatch(
            str(row["normalized_non_horizon_settings_sha256"])
        )
        or row.get("unresolved_source_fields") != []
        or row.get("fields_added_by_current_defaults") != []
        or row.get("changed_scalar_paths")
        != list(ALLOWED_PROTOCOL_DELTA_PATHS)
        or set(changed_by_path) != set(ALLOWED_PROTOCOL_DELTA_PATHS)
        or len(changed_values) != len(ALLOWED_PROTOCOL_DELTA_PATHS)
        or changed_by_path["bundle_materialization.cell_id"].get("source")
        != semantics["source_execution_id"]
        or changed_by_path["bundle_materialization.cell_id"].get("target")
        != execution_id
        or changed_by_path["horizon"].get("source") != SOURCE_HORIZON
        or changed_by_path["horizon"].get("target") != TARGET_HORIZON
        or changed_by_path[
            "request.execution.stop.maximum_controller_rounds"
        ].get("source")
        != SOURCE_HORIZON
        or changed_by_path[
            "request.execution.stop.maximum_controller_rounds"
        ].get("target")
        != TARGET_HORIZON
        or changed_by_path[
            "route_contract.execution_settings.maximum_controller_rounds"
        ].get("source")
        != SOURCE_HORIZON
        or changed_by_path[
            "route_contract.execution_settings.maximum_controller_rounds"
        ].get("target")
        != TARGET_HORIZON
        or changed_by_path[
            "stopping_rule.maximum_controller_rounds"
        ].get("source")
        != SOURCE_HORIZON
        or changed_by_path[
            "stopping_rule.maximum_controller_rounds"
        ].get("target")
        != TARGET_HORIZON
        or changed_by_path["sha256"].get("source")
        != source_protocol["canonical_sha256"]
        or changed_by_path["sha256"].get("target")
        != row["derived_protocol_sha256"]
    ):
        raise PackageContractError(
            f"Horizon-only delta row drifted: {execution_id}"
        )
    for path in (
        "bundle_materialization.sha256",
        "route_contract.sha256",
    ):
        source = changed_by_path[path].get("source")
        target = changed_by_path[path].get("target")
        if (
            not isinstance(source, str)
            or not _HEX64.fullmatch(source)
            or not isinstance(target, str)
            or not _HEX64.fullmatch(target)
            or source == target
        ):
            raise PackageContractError(
                f"Delta digest did not change at {path}: {execution_id}"
            )
    source_root = (
        f"runs/{semantics['source_execution_id']}"
    )
    target_root = f"runs/{execution_id}"
    expected_paths = {
        "request.observation.checkpoint.path": (
            f"{source_root}/checkpoints/current.json",
            f"{target_root}/checkpoints/current.json",
        ),
        "request.observation.estimator_ledger.path": (
            f"{source_root}/result/estimator_ledger.json",
            f"{target_root}/result/estimator_ledger.json",
        ),
    }
    for path, (source, target) in expected_paths.items():
        if (
            changed_by_path[path].get("source") != source
            or changed_by_path[path].get("target") != target
        ):
            raise PackageContractError(
                f"Delta output identity drifted at {path}: {execution_id}"
            )


def _validate_job(
    job: Mapping[str, Any],
    *,
    source_members: Mapping[str, Mapping[str, Any]],
    source_authority_row: Mapping[str, Any],
    delta_row: Mapping[str, Any],
) -> None:
    verify_self_digest(job, label=f"job {job.get('execution_id')}")
    execution_id = str(job.get("execution_id", ""))
    semantics = _expected_execution_semantics(execution_id)
    source_execution_id = semantics["source_execution_id"]
    route_id = semantics["route_id"]
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("run_class") != RUN_CLASS
        or job.get("execution_target") != EXECUTION_TARGET
        or job.get("source_execution_id") != source_execution_id
        or job.get("cell_id") != execution_id
        or job.get("regime_id") != semantics["regime_id"]
        or int(job.get("nph", -1)) != semantics["nph"]
        or job.get("route_id") != route_id
        or job.get("candidate_representation")
        != semantics["candidate_representation"]
        or job.get("execution_entrypoint") != "run_append_adapt"
        or job.get("source_lock_id") != semantics["source_lock_id"]
        or job.get("source_job") != source_authority_row["source_job"]
        or job.get("horizon")
        != {"source": SOURCE_HORIZON, "target": TARGET_HORIZON}
        or job.get("fresh_start_contract")
        != {
            "kind": "fresh_start",
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "controller_round_origin": 0,
        }
        or job.get("artifact_paths")
        != _expected_artifact_paths(execution_id)
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submission_state") != "not_submitted"
        or job.get("resources") != semantics["resources"]
    ):
        raise PackageContractError(f"Job semantics drifted: {execution_id}")
    source_protocol = job.get("source_protocol")
    if not isinstance(source_protocol, Mapping):
        raise PackageContractError(
            f"Job lacks a source protocol: {execution_id}"
        )
    relative = str(source_protocol.get("path", ""))
    declared = source_members.get(relative)
    if (
        declared is None
        or source_protocol != source_authority_row["source_protocol"]
        or source_protocol.get("sha256") != declared.get("sha256")
        or int(source_protocol.get("size_bytes", -1))
        != int(declared.get("size_bytes", -2))
    ):
        raise PackageContractError(
            f"Job source-protocol binding drifted: {execution_id}"
        )
    if (
        delta_row.get("execution_id") != execution_id
        or delta_row.get("source_execution_id") != source_execution_id
        or delta_row.get("source_protocol_sha256")
        != source_protocol.get("canonical_sha256")
        or delta_row.get("derived_protocol_sha256")
        != job.get("derived_protocol_sha256")
        or delta_row.get("normalized_non_horizon_settings_match")
        is not True
        or delta_row.get("unresolved_source_fields") != []
        or delta_row.get("fields_added_by_current_defaults") != []
        or delta_row.get("changed_scalar_paths")
        != list(ALLOWED_PROTOCOL_DELTA_PATHS)
    ):
        raise PackageContractError(
            f"Job/delta audit binding drifted: {execution_id}"
        )
def _manifest_file_binding(
    package_dir: Path,
    relative: str,
) -> dict[str, Any]:
    path = package_dir / relative
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Package member is unsafe: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }


def validate_package(
    *,
    package_dir: Path = PACKAGE_DIR,
    full_archive_scan: bool = False,
    full_anchor_scan: bool = False,
) -> dict[str, Any]:
    if package_dir.name != PACKAGE_ID:
        raise PackageContractError("Package directory identity drifted.")
    source_manifest = validate_source_archive(
        package_dir=package_dir,
        full_member_scan=full_archive_scan,
    )
    source_members = _archive_member_map(source_manifest)
    source_authority = load_json_object(
        package_dir / "source_authority.json",
        label="source authority",
    )
    verify_self_digest(source_authority, label="source authority")
    if (
        source_authority.get("schema") != SOURCE_AUTHORITY_SCHEMA
        or source_authority.get("status") != "passed"
        or source_authority.get("source_package_id") != SOURCE_PACKAGE_ID
        or source_authority.get("source_package_manifest_sha256")
        != SOURCE_PACKAGE_MANIFEST_SHA256
        or source_authority.get("source_package_manifest_file_sha256")
        != SOURCE_PACKAGE_MANIFEST_FILE_SHA256
        or source_authority.get("source_execution_plan_sha256")
        != SOURCE_EXECUTION_PLAN_SHA256
        or source_authority.get("source_execution_plan_file_sha256")
        != SOURCE_EXECUTION_PLAN_FILE_SHA256
        or source_authority.get("source_archive_manifest_sha256")
        != SOURCE_ARCHIVE_MANIFEST_SHA256
        or source_authority.get("source_archive_manifest_file_sha256")
        != SOURCE_ARCHIVE_MANIFEST_FILE_SHA256
        or source_authority.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or source_authority.get("source_archive_size_bytes")
        != SOURCE_ARCHIVE_SIZE_BYTES
        or source_authority.get("source_final_receipt_sha256")
        != SOURCE_FINAL_RECEIPT_SHA256
        or source_authority.get("source_final_receipt_file_sha256")
        != SOURCE_FINAL_RECEIPT_FILE_SHA256
        or source_authority.get("source_authorization_sha256")
        != SOURCE_AUTHORIZATION_SHA256
        or source_authority.get("source_authorization_file_sha256")
        != SOURCE_AUTHORIZATION_FILE_SHA256
        or source_authority.get("source_package_validation_status")
        != "passed"
        or source_authority.get("source_append_row_count")
        != DIRECT_EXECUTION_COUNT
        or source_authority.get("source_role")
        != "completed_visible_round50_settings_authority_only_v1"
        or source_authority.get("source_checkpoint_consumed") is not False
        or source_authority.get("source_result_consumed") is not False
    ):
        raise PackageContractError("Round-50 source authority drifted.")
    source_rows = source_authority.get("source_rows")
    if not isinstance(source_rows, list):
        raise PackageContractError("Source authority has no row bindings.")
    source_rows_by_id = {
        str(row.get("execution_id")): row
        for row in source_rows
        if isinstance(row, Mapping)
    }
    if (
        len(source_rows) != DIRECT_EXECUTION_COUNT
        or set(source_rows_by_id) != set(EXPECTED_EXECUTION_IDS)
    ):
        raise PackageContractError("Source-authority row set drifted.")
    for execution_id in EXPECTED_EXECUTION_IDS:
        _validate_source_authority_row(
            source_rows_by_id[execution_id],
            execution_id=execution_id,
            source_members=source_members,
        )

    anchor_evidence = load_json_object(
        package_dir / ANCHOR_EVIDENCE_NAME,
        label="independent anchor evidence",
    )
    validate_anchor_evidence(
        anchor_evidence,
        full_attempt_scan=full_anchor_scan,
    )
    audit = load_json_object(
        package_dir / "horizon_delta_audit.json",
        label="horizon delta audit",
    )
    verify_self_digest(audit, label="horizon delta audit")
    rows = audit.get("planned_rows")
    audit_source = audit.get("source")
    audit_sweep = audit.get("sweep")
    if (
        not isinstance(rows, list)
        or not isinstance(audit_source, Mapping)
        or not isinstance(audit_sweep, Mapping)
    ):
        raise PackageContractError("Horizon delta audit has no planned rows.")
    rows_by_id = {
        str(row.get("execution_id")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if (
        audit.get("schema") != DELTA_AUDIT_SCHEMA
        or audit.get("status") != "pass"
        or audit_source.get("source_horizon") != SOURCE_HORIZON
        or audit_source.get("source_variable_value") != SOURCE_HORIZON
        or audit_source.get("source_package_id") != SOURCE_PACKAGE_ID
        or audit_source.get("source_package_manifest_sha256")
        != SOURCE_PACKAGE_MANIFEST_SHA256
        or audit_source.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or audit_source.get("source_json")
        != (
            f"{SOURCE_ATTEMPT_RELATIVE_PATH}#"
            "worker_outputs/result.json"
        )
        or audit_source.get("source_sha256") != ANCHOR_RESULT_SHA256
        or audit_source.get("source_command_or_manifest")
        != (
            f"{SOURCE_PACKAGE_RELATIVE_ROOT}/jobs/"
            f"{ANCHOR_EXECUTION_ID}.json"
        )
        or audit_source.get("source_command_or_manifest_sha256")
        != SOURCE_ANCHOR_JOB_FILE_SHA256
        or audit_source.get("settings_hash") != ANCHOR_PROTOCOL_SHA256
        or audit_sweep.get("variable") != "maximum_controller_rounds"
        or audit_sweep.get("grid") != [TARGET_HORIZON]
        or audit_sweep.get("runner_mode") != "fresh_full_replay"
        or audit_sweep.get("wrapper_used") is not False
        or audit_sweep.get("unresolved_source_fields") != []
        or audit_sweep.get("fields_added_by_current_defaults") != []
        or audit_sweep.get("settings_changed")
        != ["maximum_controller_rounds", "output_identity"]
        or audit.get("anchor")
        != audit_anchor_payload(anchor_evidence)
        or set(rows_by_id) != set(EXPECTED_EXECUTION_IDS)
    ):
        raise PackageContractError("Horizon-only sensitivity audit drifted.")
    if len(rows) != DIRECT_EXECUTION_COUNT:
        raise PackageContractError("Horizon delta row count drifted.")
    for execution_id in EXPECTED_EXECUTION_IDS:
        _validate_delta_row(
            rows_by_id[execution_id],
            execution_id=execution_id,
            source_authority_row=source_rows_by_id[execution_id],
        )

    jobs: dict[str, dict[str, Any]] = {}
    for execution_id in EXPECTED_EXECUTION_IDS:
        path = package_dir / "jobs" / f"{execution_id}.json"
        job = load_json_object(path, label=f"job {execution_id}")
        _validate_job(
            job,
            source_members=source_members,
            source_authority_row=source_rows_by_id[execution_id],
            delta_row=rows_by_id[execution_id],
        )
        jobs[execution_id] = job

    plan = load_json_object(
        package_dir / "execution_plan.json", label="execution plan"
    )
    verify_self_digest(plan, label="execution plan")
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("package_id") != PACKAGE_ID
        or plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("run_class") != RUN_CLASS
        or plan.get("execution_target") != EXECUTION_TARGET
        or plan.get("source_horizon") != SOURCE_HORIZON
        or plan.get("target_horizon") != TARGET_HORIZON
        or plan.get("fresh_start") is not True
        or plan.get("resume_claimed") is not False
        or plan.get("direct_execution_count") != DIRECT_EXECUTION_COUNT
        or plan.get("execution_ids") != list(EXPECTED_EXECUTION_IDS)
        or plan.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or plan.get("source_authority_sha256")
        != source_authority["sha256"]
        or plan.get("anchor_evidence_sha256")
        != anchor_evidence["sha256"]
        or plan.get("horizon_delta_audit_sha256") != audit["sha256"]
        or plan.get("execution_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("submission_state") != "not_submitted"
        or plan.get("remote_stage") is not False
        or plan.get("condor_submit") is not False
    ):
        raise PackageContractError("Execution plan drifted.")
    planned = plan.get("direct_executions")
    if not isinstance(planned, list) or len(planned) != DIRECT_EXECUTION_COUNT:
        raise PackageContractError("Execution-plan row count drifted.")
    for row, execution_id in zip(
        planned, EXPECTED_EXECUTION_IDS, strict=True
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("execution_id") != execution_id
            or row.get("job_spec_path") != f"jobs/{execution_id}.json"
            or row.get("job_spec_sha256") != jobs[execution_id]["sha256"]
        ):
            raise PackageContractError(
                f"Execution-plan binding drifted: {execution_id}"
            )

    manifest = load_json_object(
        package_dir / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("run_class") != RUN_CLASS
        or manifest.get("execution_target") != EXECUTION_TARGET
        or manifest.get("execution_plan_sha256") != plan["sha256"]
        or manifest.get("source_authority_sha256")
        != source_authority["sha256"]
        or manifest.get("anchor_evidence_sha256")
        != anchor_evidence["sha256"]
        or manifest.get("horizon_delta_audit_sha256") != audit["sha256"]
        or manifest.get("source_archive")
        != {
            "path": SOURCE_ARCHIVE_NAME,
            "sha256": SOURCE_ARCHIVE_SHA256,
            "size_bytes": SOURCE_ARCHIVE_SIZE_BYTES,
        }
        or manifest.get("direct_execution_count")
        != DIRECT_EXECUTION_COUNT
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_state") != "not_submitted"
        or manifest.get("activation_required_before_submission") is not True
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
    ):
        raise PackageContractError("Package manifest drifted.")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise PackageContractError("Package manifest has no file inventory.")
    file_rows = {
        str(row.get("path")): row
        for row in raw_files
        if isinstance(row, Mapping)
    }
    expected_files = {
        *STATIC_CONTROL_FILES,
        *GENERATED_CONTROL_FILES,
        *(f"jobs/{execution_id}.json" for execution_id in EXPECTED_EXECUTION_IDS),
    }
    if set(file_rows) != expected_files:
        raise PackageContractError("Package manifest file set drifted.")
    for relative in sorted(expected_files):
        if file_rows[relative] != _manifest_file_binding(
            package_dir, relative
        ):
            raise PackageContractError(
                f"Package file binding drifted: {relative}"
            )
    actual_files = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.startswith(".")
    }
    if actual_files != expected_files | {"package_manifest.json"}:
        raise PackageContractError("On-disk package file set drifted.")
    if any(
        (package_dir / name).exists()
        for name in ("submit.sub", "execute_source_locked_job.sh", "authority")
    ):
        raise PackageContractError(
            "Inert package unexpectedly contains an activation surface."
        )
    return {
        "status": "passed",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "execution_plan_sha256": plan["sha256"],
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "anchor_evidence_sha256": anchor_evidence["sha256"],
        "anchor_reproduces_source": True,
        "anchor_reproduction_scope": "scientific_payload_only",
        "anchor_independent_execution": True,
        "anchor_worker_exit_status": ANCHOR_WORKER_EXIT_STATUS,
        "anchor_full_worker_attempt_passed": False,
        "source_worker_exit_status": SOURCE_WORKER_EXIT_STATUS,
        "source_full_worker_attempt_passed": True,
        "horizon_delta_audit_sha256": audit["sha256"],
        "direct_execution_count": DIRECT_EXECUTION_COUNT,
        "source_horizon": SOURCE_HORIZON,
        "target_horizon": TARGET_HORIZON,
        "fresh_start": True,
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_state": "not_submitted",
        "activation_required_before_submission": True,
        "remote_stage": False,
        "condor_submit": False,
    }


def validate_execution_authorization(
    authorization_path: Path,
    *,
    execution_id: str,
    package_dir: Path = PACKAGE_DIR,
) -> dict[str, Any]:
    package = validate_package(
        package_dir=package_dir, full_archive_scan=False
    )
    authorization = load_json_object(
        authorization_path, label="execution authorization"
    )
    verify_self_digest(authorization, label="execution authorization")
    if (
        authorization.get("schema") != EXECUTION_AUTHORIZATION_SCHEMA
        or authorization.get("status") != "passed"
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("package_manifest_sha256")
        != package["package_manifest_sha256"]
        or authorization.get("authorized_execution_ids")
        != list(EXPECTED_EXECUTION_IDS)
        or execution_id not in authorization["authorized_execution_ids"]
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("submission_state")
        != "authorized_not_submitted"
    ):
        raise PackageContractError("Execution authorization drifted.")
    return authorization


__all__ = [
    "ALLOWED_PROTOCOL_DELTA_PATHS",
    "ANCHOR_EVIDENCE_NAME",
    "ANCHOR_EVIDENCE_SCHEMA",
    "ANCHOR_EXECUTION_ID",
    "ANCHOR_WORKER_EXIT_STATUS",
    "CAMPAIGN_ID",
    "CANDIDATE_REPRESENTATIONS",
    "DELTA_AUDIT_SCHEMA",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_AUTHORIZATION_SCHEMA",
    "EXECUTION_PLAN_SCHEMA",
    "EXPECTED_ARTIFACT_ROLES",
    "EXPECTED_EXECUTION_IDS",
    "EXPECTED_SOURCE_EXECUTION_IDS",
    "GENERATED_CONTROL_FILES",
    "GENERATED_DIRECTORIES",
    "JOB_SCHEMA",
    "MUTABLE_RUNTIME_DIRECTORIES",
    "PACKAGE_DIR",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_SCHEMA",
    "PACKAGE_RELATIVE_ROOT",
    "PackageContractError",
    "REPO_ROOT",
    "RUN_CLASS",
    "SOURCE_ARCHIVE_MANIFEST_FILE_SHA256",
    "SOURCE_ARCHIVE_MANIFEST_NAME",
    "SOURCE_ARCHIVE_MANIFEST_SHA256",
    "SOURCE_ARCHIVE_NAME",
    "SOURCE_ARCHIVE_SHA256",
    "SOURCE_ARCHIVE_SIZE_BYTES",
    "SOURCE_AUTHORITY_SCHEMA",
    "SOURCE_BUNDLE_RELATIVE_ROOT",
    "SOURCE_EXECUTION_PLAN_FILE_SHA256",
    "SOURCE_EXECUTION_PLAN_SHA256",
    "SOURCE_FINAL_RECEIPT_FILE_SHA256",
    "SOURCE_FINAL_RECEIPT_SHA256",
    "SOURCE_HORIZON",
    "SOURCE_PACKAGE_DIR",
    "SOURCE_PACKAGE_ID",
    "SOURCE_PACKAGE_MANIFEST_FILE_SHA256",
    "SOURCE_PACKAGE_MANIFEST_SHA256",
    "STATIC_CONTROL_FILES",
    "SOURCE_WORKER_EXIT_STATUS",
    "TARGET_HORIZON",
    "audit_anchor_payload",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "load_json_object",
    "materialize_anchor_evidence",
    "safe_relative_path",
    "sha256_file",
    "validate_execution_authorization",
    "validate_anchor_evidence",
    "validate_package",
    "validate_source_archive",
    "verify_self_digest",
]
