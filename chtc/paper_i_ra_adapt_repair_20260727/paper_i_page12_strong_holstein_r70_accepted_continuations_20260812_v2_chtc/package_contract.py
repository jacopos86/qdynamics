#!/usr/bin/env python3
"""Closed contract for Page-12 strong-sector CHTC continuations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping

from vendor.ijson_pure import common as streaming_json_common
from vendor.ijson_pure.backends import python as streaming_json


PACKAGE_ID = (
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260812_v2_chtc"
)
CAMPAIGN_ID = "paper_i_page12_strong_holstein_r70_continuations_20260812_v2"
BUNDLE_ID = "paper_i_page12_strong_holstein_r70_continuations_20260812_v2"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70
ROUTE_ID = (
    "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_"
    "phase3_qiskit_phase2_phase3_plateau_no_lanes_v1"
)
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_candidate_adapter_v1"
)
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__global_singleton_abs_gradient_"
    "phase0_then_singleton_phase1_then_qiskit_phase2_phase3_no_lanes_v1"
)
ROUTE_CONTRACT_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_OUTPUT_ROOT = (
    "/staging/jsstrobel/"
    "paper_i_page12_strong_r70_continuations_20260812_v2/outputs"
)
BACKEND_COMPILE_SCOPE = (
    "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
)
SELECTOR_COMPILE_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
)
PHASE0_POLICY = "global_singleton_absolute_gradient_shortlist_v1"
PHASE0_SHORTLIST_SIZE = 24
EXPECTED_CANDIDATE_FUNNEL = (
    "global_singleton_gradient_phase0_shortlist_then_singleton_phase1_"
    "shortlist_then_singleton_phase2_then_singleton_phase3_v1"
)

BASE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_r50_20260807_v1_chtc"
)
BASE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "ae96ea800ac108b207e4ccdac148584f1bc5dd6082dd23da893b9f958c1a1896"
)
BASE_PACKAGE_MANIFEST_SHA256 = (
    "a0930b878087799aa81b37b5dcaf8a66859aebf871f4e4e02054fa58a82f6731"
)
BASE_SOURCE_ARCHIVE_SHA256 = (
    "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
)
BASE_SOURCE_MANIFEST_FILE_SHA256 = (
    "5aedd26f1578ca56e214ee210cc2e8a3e6eab9b40f3bb9d787359438b692e1f8"
)
BASE_SOURCE_MANIFEST_SHA256 = (
    "0470584463090ffa732b9ddbd4dd016781a0cbf1b8c31cec120acdc7afd8cddf"
)
BASE_SOURCE_LOCKS_FILE_SHA256 = (
    "1dae0adc61161be5e3fbda22c9d0c035f4da2f4a946c011cfaf13ed5d7b2ab99"
)
BASE_SOURCE_LOCKS_SHA256 = (
    "e8da64fc347cba75ba733434c2c8cc46142875ad46a77fc91fdc9500c5ab2ae6"
)
V1_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260811_v1_chtc"
)
V1_PACKAGE_ID = (
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260811_v1_chtc"
)
V1_PACKAGE_MANIFEST_FILE_SHA256 = (
    "926f0969e142d964744d7a130ce5568df9751b4e683945559e2586709ae90884"
)
V1_PACKAGE_MANIFEST_SHA256 = (
    "3051aa31402d6c71d87ec7ca9d12006ba95fd95ae22d61ff679110578de2671b"
)
V1_PACKAGE_MANIFEST_SIZE_BYTES = 9944
CONTROLLER_RELATIVE_PATH = "pipelines/static_adapt/sr_snake/_controller.py"
CONTROLLER_BEFORE_SHA256 = (
    "4b0171acdbf9b0c9bcf3d0d626da959ffae7f20129eaa2ee251d02f62c8c265a"
)
CONTROLLER_AFTER_SHA256 = (
    "e25c0281373b828f75200410aa0e5364eaebe5a78f517421bc8c7bdc73c20327"
)
CONTROLLER_REPAIR_ID = "accepted_energy_roundoff_only_128ulp_v1"
CONTROLLER_REGRESSION = (
    "test/test_static_adapt_sr_snake_controller.py::"
    "test_selection_state_accepts_only_roundoff_scale_energy_replay"
)
RESUME_RELATIVE_PATH = "pipelines/static_adapt/sr_snake/_resume.py"
RESUME_BEFORE_SHA256 = (
    "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
)
RESUME_AFTER_SHA256 = (
    "00a06606cf69dce5ee749172839b2115a5b5bb7dce72b170fc58d792ee1d79a6"
)
RESUME_AFTER_SIZE_BYTES = 202772
RESUME_REPAIR_ID = "phase0_gradient_screen_resume_closure_v1"
RESUME_REGRESSION = (
    "test/test_static_adapt_resume_insertion_integrity.py::"
    "test_resume_accepts_authenticated_phase0_shortlist_as_phase_i"
)

BASE_PROTOCOL_ROOT = (
    BASE_PACKAGE_RELATIVE
    / "bundle_materialization/"
    "ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
    "no_lanes_cap24_tau1em4_r50_v1/protocols"
)


def source_execution_id(regime_id: str) -> str:
    return (
        "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
        f"{regime_id}__nph7__{ROUTE_ID}"
    )


def execution_id(regime_id: str) -> str:
    return f"page12_r70_resume__{regime_id}__nph7__{ROUTE_ID}"


CELL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "regime_id": "intermediate_strong",
        "resume_round": 50,
        "source_protocol_sha256": "6a325b38caa74ffd484d08a804996c317968900739e7d4ca64493cb3531b78d0",
        "source_job_file_sha256": "64e04b8a8adf21f2e7b61a04e05446c50f7b66c8e8cf8aea701c0aabd120217a",
        "source_job_sha256": "8a8a252669e6fc1a20284672a66cbf25c148415472d1fdfea4cc56fc8696165c",
        "v1_resume_archive": {"sha256": "c3c86df45c7547c5cd6dd19439aac36ce6354ef999f3c33e6c8a911b6caa7bcc", "size_bytes": 827002429},
        "v1_resume_manifest": {"sha256": "4c15a929503f66542a48f9548ae4b247468d5fa07d43206151830d5ea8e193de", "canonical_sha256": "03384eb3168c669e80cf585f26d50091f2ecde80ade87ed76bdfcb99e9862595", "size_bytes": 4097},
        "v1_checkpoint_validation": {"sha256": "3eae455f04d10522bcbe6fcae95a020e7c11da2a393ac5f3c82e2f9ebcc91d96", "canonical_sha256": "1f79d1a15270d0e8f755f62ecc513a8534a1b628f8e47fab70c18b620cfa5416", "size_bytes": 5647},
    },
    {
        "regime_id": "strong_strong_u8",
        "resume_round": 50,
        "source_protocol_sha256": "ffe0d3128448c71666f44b7ccd0abe48673e513ce30c703dbf194331d84e8849",
        "source_job_file_sha256": "cff337847e4704e75038f1f79b1b56e5fd5dbc99f66b529eb4d6a538c5c9222c",
        "source_job_sha256": "585ac5314cd0b0ab6d75c822f90b476eccbe5415854d46cd6b1745ed8d63fba3",
        "v1_resume_archive": {"sha256": "3d4ceec0c0383537442cae6bd8d6c9b1d9d79339f90a7431ae282dfd45716710", "size_bytes": 902166171},
        "v1_resume_manifest": {"sha256": "916c0febfd4df4e40b0214f49f891a364b2026018b3ccc89494d172b1a961e1a", "canonical_sha256": "4e1520a238f2e20f35a5808ed6e34b572e902e135794e92a84fe7a6c660e64e4", "size_bytes": 4067},
        "v1_checkpoint_validation": {"sha256": "284d0301c39133f4659dc8fc61d1a652c7389c3b8887b7fbad1ce67fb18a139c", "canonical_sha256": "d0654da0c2d92903953a3dd45226fa14acb36a605ae76e0f82042a14fe44c78c", "size_bytes": 5618},
    },
)
CONTINUATION_ROW_COUNT = len(CELL_SPECS)

RESOURCE_ENVELOPE = {
    "request_cpus": 4,
    "request_memory_mb": 131072,
    "request_disk_mb": 102400,
    "max_runtime_seconds": 259200,
    "basis": (
        "page12_strong_sector_cgroup_memory_repair_uniform_128g_v2"
    ),
}

PACKAGE_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_chtc_package_v2"
JOB_SCHEMA = "paper_i_page12_strong_r70_chtc_job_v2"
BUNDLE_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_chtc_bundle_v2"
ACTIVATION_SCHEMA = "paper_i_page12_strong_r70_chtc_activation_v2"
AUTHORIZATION_SCHEMA = "paper_i_page12_strong_r70_chtc_authorization_v2"

VENDORED_STREAMING_JSON_FILES = (
    "vendor/__init__.py",
    "vendor/ijson_pure/__init__.py",
    "vendor/ijson_pure/LICENSE.txt",
    "vendor/ijson_pure/common.py",
    "vendor/ijson_pure/compat.py",
    "vendor/ijson_pure/utils.py",
    "vendor/ijson_pure/utils35.py",
    "vendor/ijson_pure/backends/__init__.py",
    "vendor/ijson_pure/backends/python.py",
)
VENDORED_STREAMING_JSON_VERSION = "3.5.1"
VENDORED_STREAMING_JSON_BACKEND = "python"

CONTROL_FILES = (
    "package_contract.py",
    "derive_protocol.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
    "execute_authorized_job.sh",
    *VENDORED_STREAMING_JSON_FILES,
)


class PackageContractError(RuntimeError):
    """Fail-closed continuation package error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    observed = unsigned.pop("sha256", None)
    expected = canonical_sha256(unsigned)
    if observed != expected:
        raise PackageContractError(f"{label} self-digest drifted.")
    return expected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a nonempty path.")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise PackageContractError(f"{label} is unsafe: {value!r}")
    return Path(*path.parts)


def repo_root_from_script(script: str | Path) -> Path:
    current = Path(script).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "pipelines/static_adapt").is_dir():
            return candidate
    raise PackageContractError("Could not resolve active repository root.")


def file_binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PackageContractError(f"Missing or unsafe file: {path}")
    relative = resolved.relative_to(root.resolve()).as_posix()
    result: dict[str, Any] = {"path": relative, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}
    if canonical:
        result["canonical_sha256"] = verify_self_digest(load_json(resolved, label=relative), label=relative)
    return result


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(execution_id(str(row["regime_id"])) for row in CELL_SPECS)


def _checkpoint_metadata(stream: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"active_prefix_checkpoint_count": 0}
    ledger: dict[str, Any] = {}
    resume: dict[str, Any] = {}
    for prefix, event, value in streaming_json.parse(stream):
        if prefix == "adapt_vqe.active_prefix_checkpoints.item" and event == "start_map":
            metadata["active_prefix_checkpoint_count"] += 1
        elif prefix == "checkpoint.depth" and event in {"integer", "number"}:
            metadata["checkpoint_depth"] = int(value)
        elif prefix == "adapt_vqe.history_count" and event in {"integer", "number"}:
            metadata["history_count"] = int(value)
        elif prefix == "adapt_vqe.history_checkpoint_complete" and event == "boolean":
            metadata["history_checkpoint_complete"] = value
        elif prefix == "adapt_vqe.strict_replay.passed" and event == "boolean":
            metadata["strict_replay_passed"] = value
        elif prefix == "adapt_vqe.energy" and event in {"integer", "number"}:
            metadata["checkpoint_energy"] = float(value)
        elif (
            prefix
            == "adapt_vqe.terminal_active_prefix_checkpoint."
            "post_admission_prune.energy_after_post_refit"
            and event in {"integer", "number"}
        ):
            metadata["accepted_prefix_terminal_energy"] = float(value)
        elif (
            prefix
            == "adapt_vqe.terminal_active_prefix_checkpoint."
            "projective_state_fingerprint"
            and event == "string"
        ):
            metadata["accepted_prefix_terminal_state_fingerprint"] = value
        elif prefix == "adapt_vqe.sr_route_profile_contract_sha256" and event == "string":
            metadata["route_contract_sha256"] = value
        elif prefix.startswith("adapt_vqe.estimator_call_ledger_checkpoint.") and event in {"boolean", "integer", "number", "string"}:
            ledger[prefix.rsplit(".", 1)[-1]] = value
        elif prefix.startswith("adapt_vqe.verified_singleton_resume_sidecar.") and event in {"boolean", "integer", "number", "string"}:
            resume[prefix.rsplit(".", 1)[-1]] = value
    metadata["estimator_call_ledger_checkpoint"] = ledger
    metadata["verified_singleton_resume_sidecar"] = resume
    return metadata


def validate_resume_archive(
    archive_path: Path,
    manifest: Mapping[str, Any],
    *,
    expected_round: int,
    checkpoint_validation: Mapping[str, Any] | None = None,
    verify_archive_members: bool = True,
) -> dict[str, Any]:
    """Authenticate an exact pointer-closed three-member resume archive."""

    verify_self_digest(manifest, label="resume archive manifest")
    archive = manifest.get("archive")
    raw_members = manifest.get("members")
    if not isinstance(archive, Mapping) or not isinstance(raw_members, list):
        raise PackageContractError("Resume archive manifest is incomplete.")
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size != int(archive.get("size_bytes", -1))
        or sha256_file(archive_path) != archive.get("sha256")
        or len(raw_members) != 3
        or manifest.get("member_count") != 3
        or manifest.get("pointer_closed") is not True
        or manifest.get("resume_round") != expected_round
    ):
        raise PackageContractError("Resume archive binding drifted.")
    by_path: dict[str, Mapping[str, Any]] = {}
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw in raw_members:
        if not isinstance(raw, Mapping):
            raise PackageContractError("Resume member is malformed.")
        relative = safe_relative_path(raw.get("path"), label="resume member").as_posix()
        role = str(raw.get("role", ""))
        if relative in by_path or role in by_role:
            raise PackageContractError("Resume member identity is duplicated.")
        by_path[relative] = raw
        by_role[role] = raw
    if set(by_role) != {"checkpoint", "estimator_ledger_checkpoint", "verified_resume_sidecar"}:
        raise PackageContractError("Resume member roles are not closed.")
    try:
        if verify_archive_members:
            observed: set[str] = set()
            with tarfile.open(archive_path, "r:gz") as opened:
                for member in opened:
                    relative = safe_relative_path(member.name, label="resume tar member").as_posix()
                    row = by_path.get(relative)
                    if (
                        row is None
                        or relative in observed
                        or not member.isfile()
                        or member.issym()
                        or member.islnk()
                        or member.size != int(row.get("size_bytes", -1))
                    ):
                        raise PackageContractError(f"Unsafe resume member: {relative}")
                    source = opened.extractfile(member)
                    if source is None:
                        raise PackageContractError(f"Unreadable resume member: {relative}")
                    digest = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                    if size != member.size or digest.hexdigest() != row.get("sha256"):
                        raise PackageContractError(f"Resume member drifted: {relative}")
                    observed.add(relative)
            if observed != set(by_path):
                raise PackageContractError("Resume member closure is incomplete.")
        checkpoint_row = by_role["checkpoint"]
        if checkpoint_validation is None:
            if not verify_archive_members:
                raise PackageContractError(
                    "Checkpoint metadata cannot be minted without member validation."
                )
            with tarfile.open(archive_path, "r:gz") as opened:
                checkpoint_member = opened.getmember(str(checkpoint_row["path"]))
                checkpoint_stream = opened.extractfile(checkpoint_member)
                if checkpoint_stream is None:
                    raise PackageContractError("Resume checkpoint is unreadable.")
                metadata = _checkpoint_metadata(checkpoint_stream)
            validation_source = "vendored_pure_python_stream_parse_v1"
        else:
            verify_self_digest(
                checkpoint_validation,
                label="checkpoint validation receipt",
            )
            if (
                checkpoint_validation.get("schema")
                != "paper_i_page12_checkpoint_validation_receipt_v2"
                or checkpoint_validation.get("status") != "passed"
                or checkpoint_validation.get("package_id") != PACKAGE_ID
                or checkpoint_validation.get("regime_id")
                != manifest.get("regime_id")
                or checkpoint_validation.get("resume_round") != expected_round
                or checkpoint_validation.get("archive") != archive
                or checkpoint_validation.get("members") != raw_members
                or checkpoint_validation.get("checkpoint_sha256")
                != checkpoint_row.get("sha256")
                or checkpoint_validation.get("validation_authority")
                != "sealed_v1_full_stream_validation_plus_v2_byte_identity_v1"
                or checkpoint_validation.get("worker_validation_scope")
                != "stream_authenticate_all_three_members_then_strict_resume_replay_v1"
                or checkpoint_validation.get("inherited_v1_authority")
                != manifest.get("inherited_v1_authority")
            ):
                raise PackageContractError(
                    "Checkpoint validation authority drifted."
                )
            raw_metadata = checkpoint_validation.get("metadata")
            if not isinstance(raw_metadata, Mapping):
                raise PackageContractError(
                    "Checkpoint validation metadata is absent."
                )
            metadata = dict(raw_metadata)
            validation_source = (
                "sealed_v1_full_stream_validation_plus_v2_byte_identity_v1"
            )
    except (
        OSError,
        EOFError,
        tarfile.TarError,
        streaming_json_common.JSONError,
    ) as exc:
        raise PackageContractError("Resume archive is invalid.") from exc
    ledger = metadata.get("estimator_call_ledger_checkpoint")
    sidecar = metadata.get("verified_singleton_resume_sidecar")
    if not isinstance(ledger, Mapping) or not isinstance(sidecar, Mapping):
        raise PackageContractError("Checkpoint resume pointers are absent.")
    for role, pointer in (
        ("estimator_ledger_checkpoint", ledger),
        ("verified_resume_sidecar", sidecar),
    ):
        row = by_role[role]
        if (
            pointer.get("status") != "complete"
            or pointer.get("sha256") != row.get("sha256")
            or PurePosixPath(str(pointer.get("path", ""))).name
            != PurePosixPath(str(row.get("path", ""))).name
            or (pointer.get("size_bytes") is not None and int(pointer["size_bytes"]) != int(row.get("size_bytes", -1)))
            or (role == "verified_resume_sidecar" and pointer.get("enabled") is not True)
        ):
            raise PackageContractError(f"Checkpoint {role} pointer drifted.")
    if (
        metadata.get("checkpoint_depth") != expected_round
        or metadata.get("history_count") != expected_round
        or metadata.get("active_prefix_checkpoint_count") != expected_round
        or metadata.get("history_checkpoint_complete") is not True
        or metadata.get("strict_replay_passed") is not True
        or metadata.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or metadata.get("checkpoint_energy") is None
        or metadata.get("accepted_prefix_terminal_energy") is None
        or metadata.get("accepted_prefix_terminal_state_fingerprint") is None
    ):
        raise PackageContractError(
            f"Resume checkpoint is not the authenticated round-{expected_round} prefix."
        )
    return {
        "metadata": metadata,
        "members_by_role": by_role,
        "checkpoint_validation_source": validation_source,
    }
