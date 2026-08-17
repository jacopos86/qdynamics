#!/usr/bin/env python3
"""Closed contract for local Page-12 strong-sector continuations."""

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
    "20260810_v1_local"
)
CAMPAIGN_ID = "paper_i_page12_strong_holstein_r70_continuations_v1_local"
BUNDLE_ID = "paper_i_page12_strong_holstein_r70_continuations_v1_local"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "local_mac_serial"
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


def _attempt_triplet(
    regime_id: str,
    checkpoint: tuple[str, int],
    ledger: tuple[str, int],
    sidecar: tuple[str, int],
) -> tuple[dict[str, Any], ...]:
    root = f"./runs/{source_execution_id(regime_id)}/checkpoints"
    return (
        {"role": "checkpoint", "source_member": f"{root}/current.json", "archive_path": "resume/current.json", "sha256": checkpoint[0], "size_bytes": checkpoint[1]},
        {"role": "estimator_ledger_checkpoint", "source_member": f"{root}/current.estimator_call_ledger_checkpoint.{ledger[0][:16]}.json", "archive_path": f"resume/current.estimator_call_ledger_checkpoint.{ledger[0][:16]}.json", "sha256": ledger[0], "size_bytes": ledger[1]},
        {"role": "verified_resume_sidecar", "source_member": f"{root}/current.verified_singleton_resume.{sidecar[0][:16]}.json", "archive_path": f"resume/current.verified_singleton_resume.{sidecar[0][:16]}.json", "sha256": sidecar[0], "size_bytes": sidecar[1]},
    )


CELL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "regime_id": "weak_strong",
        "resume_round": 50,
        "source_protocol_sha256": "90d0a55461335ce897c9eaebb02e8c54b34c3c100d18c3a5c4f66e0d44f79225",
        "source_job_file_sha256": "2dc8d53e994cc07d05151e5803776d958b0ace34f9b7692988be7f2a46704cd5",
        "source_job_sha256": "648e84fa42da019e943939bad3244f2c13deb4c9a6cd0ac534be5ac5f6aba71f",
        "source_archive": {"path": "chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.3_full.tar.gz", "sha256": "ab3096dd82ae499fced09ebbcc462e93fa83195a1183eb532366a4a79cd19429", "size_bytes": 1099219486},
        "source_members": _attempt_triplet("weak_strong", ("f803ea2b1d744cec09be9fea0333dec876dc568ef2c8f248d38435ecf4aa83c9", 3453886467), ("c06ef8e9c712551380ed105cf53cc0681dbf3bfee56e2a244d5b4186cdbf6f65", 3028093880), ("a5e67c7f249bfabbd21027d77112c14d5ee9193180b3cb86eeb420cb674a84b7", 5261)),
    },
    {
        "regime_id": "intermediate_strong",
        "resume_round": 50,
        "source_protocol_sha256": "6a325b38caa74ffd484d08a804996c317968900739e7d4ca64493cb3531b78d0",
        "source_job_file_sha256": "64e04b8a8adf21f2e7b61a04e05446c50f7b66c8e8cf8aea701c0aabd120217a",
        "source_job_sha256": "8a8a252669e6fc1a20284672a66cbf25c148415472d1fdfea4cc56fc8696165c",
        "source_archive": {"path": "chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.4_full.tar.gz", "sha256": "703d805f27b179f483329ea8bcb8b78bd3c1a28c56f15cd3335e14ccfc896978", "size_bytes": 1169324852},
        "source_members": _attempt_triplet("intermediate_strong", ("313107ad03f2fd4e3d6bfd8dba20140845d2c961e747660edf6315b841544a29", 4773038672), ("4922aabdfedfd8f1d811159f2c4e1124e5b85ea60d4431aa7b64a1ae74b8b675", 3698158465), ("7753bf6ccaad08e43b0be7a33788402662ab09c2150babf369827e744d99f9a7", 5232)),
    },
    {
        "regime_id": "strong_strong_u8",
        "resume_round": 50,
        "source_protocol_sha256": "ffe0d3128448c71666f44b7ccd0abe48673e513ce30c703dbf194331d84e8849",
        "source_job_file_sha256": "cff337847e4704e75038f1f79b1b56e5fd5dbc99f66b529eb4d6a538c5c9222c",
        "source_job_sha256": "585ac5314cd0b0ab6d75c822f90b476eccbe5415854d46cd6b1745ed8d63fba3",
        "source_archive": {"path": "chtc/paper_i_ra_adapt_repair_20260727/retrieved_phase0_completed_20260809/9605157.5_full.tar.gz", "sha256": "dbcd8357234f2ea1ad52319b8e5da60d4e6f65c5d0d712c5f689f885675c1a70", "size_bytes": 1299721977},
        "source_members": _attempt_triplet("strong_strong_u8", ("77d4e109956c56e869c0a20453f815a1934f3a2a33542267a3ae7dbd44d015c1", 7366270843), ("1dc92920b0396e6fb71b40053deb61ec2cef53dfc4d831d167b0025105d562d2", 5011057756), ("1e9870fbcee3d5769026403931f1d01af275d90ff36b09a879e5b1ea70228f6d", 5225)),
    },
)

RESOURCE_ENVELOPE = {
    "request_cpus": 4,
    "request_memory_mb": 90112,
    "request_disk_mb": 102400,
    "max_runtime_seconds": 259200,
    "basis": (
        "page12_strong_sector_observed_memory_up_to_73gb_plus_"
        "accepted_resume_hydration_headroom_v1"
    ),
}

PACKAGE_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_local_package_v1"
JOB_SCHEMA = "paper_i_page12_strong_r70_local_job_v1"
BUNDLE_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_local_bundle_v1"
ACTIVATION_SCHEMA = "paper_i_page12_strong_r70_local_activation_v1"
AUTHORIZATION_SCHEMA = "paper_i_page12_strong_r70_local_authorization_v1"

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
    "local_serial.py",
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
    observed: set[str] = set()
    try:
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
                != "paper_i_page12_checkpoint_validation_receipt_v1"
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
                != "vendored_pure_python_full_stream_validation_v1"
                or checkpoint_validation.get("worker_validation_scope")
                != "stream_authenticate_all_three_members_then_strict_resume_replay_v1"
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
            validation_source = "vendored_pure_python_full_stream_validation_v1"
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
