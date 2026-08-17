#!/usr/bin/env python3
"""Closed contract for Page-10 strong-Holstein accepted-state continuations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping

from vendor.ijson_pure import common as streaming_json_common
from vendor.ijson_pure.backends import python as streaming_json


PACKAGE_ID = (
    "paper_i_page10_strong_holstein_r70_accepted_continuations_"
    "20260809_v2_chtc"
)
CAMPAIGN_ID = "paper_i_page10_strong_holstein_r70_continuations_v2"
BUNDLE_ID = "paper_i_page10_strong_holstein_r70_continuations_v2"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70
ROUTE_ID = "ra_macro_then_singleton_phase123_qiskit_phase23_plateau"
ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_phase1_singleton_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_then_singleton_phase_i_candidate_adapter_v1"
)
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "macro_phase1_then_singleton_phase1_then_qiskit_phase2_phase3_"
    "no_lanes_v1"
)
ROUTE_CONTRACT_SHA256 = (
    "83b5e5cb17bdfbfc8e8efb22a586d952b3343f430de15ffb58550082d17e3cf0"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
REMOTE_OUTPUT_ROOT = (
    "/staging/jsstrobel/"
    "paper_i_page10_strong_r70_continuations_20260809_v2/outputs"
)

BASE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_then_singleton_phase123_qiskit_phase23_"
    "no_lanes_tau1em4_r50_20260807_v1_chtc"
)
BASE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "a649c99703bcc433b96c0e8d6316ce8e1cc37fd0bb811a5864b8eef4c70379af"
)
BASE_PACKAGE_MANIFEST_SHA256 = (
    "260bb30c731a3dfc68f8c8a23e91d55f869cb69f740880118f4093f75b925b0e"
)
BASE_SOURCE_ARCHIVE_SHA256 = (
    "3ec3e0055d724a21b26b36fe243bdbb7d2cedab409d77278148738622c6f4605"
)
BASE_SOURCE_MANIFEST_FILE_SHA256 = (
    "4a97429b5c8076c109844af37e41f7fb3bce8064b278d3953aec76513490a7a0"
)
BASE_SOURCE_MANIFEST_SHA256 = (
    "84ce27f8f71d25fadc8563e0c8b6879b6bb8b72c36c595da33ac8ca66493cae4"
)
BASE_SOURCE_LOCKS_FILE_SHA256 = (
    "c10e91e1cc787a0107fe186aa7021ba6345bfa1f9ae1c09490d461db33a97003"
)
BASE_SOURCE_LOCKS_SHA256 = (
    "c73b912839d06f80bd117fcd1bd4c08df0d601bd9f760ccacfc598db18850ffd"
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

VISIBLE_ADAPTER_RELATIVE = Path(
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_"
    "evolving_macro_then_singleton_phase23_qiskit_no_lanes_page10_adapter.json"
)
VISIBLE_ADAPTER_FILE_SHA256 = (
    "59e910a7d616c3b3c643a5f4896a0b2f2e4dd9f7ba9f821aadbda917b3cfa9af"
)
VISIBLE_ADAPTER_SHA256 = (
    "29f97a4a513c24e3cac0fa332d581281ad16a535bc3f15bf4b45a71357471443"
)
RECOVERABLE_PREFIX_MANIFEST_RELATIVE = Path(
    "output/pdf/paper_i_stationary_vs_paper_i_route_comparison_20260729/"
    "paper_i_page10_recoverable_continuations_20260808.json"
)
RECOVERABLE_PREFIX_MANIFEST_SHA256 = (
    "231357b7e6f71affed31f9563e2cb9294629a16a17612b4eee8b3de36c7bdf67"
)
V1_CONTINUATION_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page10_strong_holstein_r70_accepted_continuations_"
    "20260809_v1_chtc"
)
V1_CONTINUATION_MANIFEST_FILE_SHA256 = (
    "12859a95ddd5538f2c99534c267e47367c29f15ecf11f0332615b73dd4d45334"
)
V1_CONTINUATION_MANIFEST_SHA256 = (
    "0ca8dff2bf145d55bdd5d2ca6975935ae7e4448ebe4abfa76566e4c673910b46"
)

BASE_PROTOCOL_ROOT = (
    BASE_PACKAGE_RELATIVE
    / "bundle_materialization"
    / "ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_"
    "tau1em4_r50_v1"
    / "protocols"
)


def source_execution_id(regime_id: str) -> str:
    return (
        "staged_phase23_qiskit_no_lanes__"
        f"{regime_id}__nph7__{ROUTE_ID}"
    )


def execution_id(regime_id: str) -> str:
    return f"page10_r70_resume__{regime_id}__nph7__{ROUTE_ID}"


def _local_triplet(root: str, checkpoint: tuple[str, int], ledger: tuple[str, int], sidecar: tuple[str, int]) -> tuple[dict[str, Any], ...]:
    return (
        {"role": "checkpoint", "source_path": f"{root}/current.json", "archive_path": "resume/current.json", "sha256": checkpoint[0], "size_bytes": checkpoint[1]},
        {"role": "estimator_ledger_checkpoint", "source_path": f"{root}/current.estimator_call_ledger_checkpoint.{ledger[0][:16]}.json", "archive_path": f"resume/current.estimator_call_ledger_checkpoint.{ledger[0][:16]}.json", "sha256": ledger[0], "size_bytes": ledger[1]},
        {"role": "verified_resume_sidecar", "source_path": f"{root}/current.verified_singleton_resume.{sidecar[0][:16]}.json", "archive_path": f"resume/current.verified_singleton_resume.{sidecar[0][:16]}.json", "sha256": sidecar[0], "size_bytes": sidecar[1]},
    )


def _v1_checkpoint_validation(
    *,
    regime_id: str,
    manifest_file_sha256: str,
    manifest_sha256: str,
    archive_sha256: str,
    archive_size_bytes: int,
    round_: int,
    ledger_sha256: str,
    ledger_path: str,
    ledger_fingerprint: str,
    s_alg: int,
    s_unique: int,
    sidecar_sha256: str,
    sidecar_path: str,
    source_projection_sha256: str,
) -> dict[str, Any]:
    return {
        "source_package_manifest": {
            "path": (
                V1_CONTINUATION_PACKAGE_RELATIVE / "package_manifest.json"
            ).as_posix(),
            "sha256": V1_CONTINUATION_MANIFEST_FILE_SHA256,
            "canonical_sha256": V1_CONTINUATION_MANIFEST_SHA256,
        },
        "source_resume_manifest": {
            "path": (
                V1_CONTINUATION_PACKAGE_RELATIVE
                / "resume_inputs"
                / f"{regime_id}.manifest.json"
            ).as_posix(),
            "sha256": manifest_file_sha256,
            "canonical_sha256": manifest_sha256,
        },
        "archive": {
            "path": f"resume_inputs/{regime_id}.tar.gz",
            "sha256": archive_sha256,
            "size_bytes": archive_size_bytes,
        },
        "metadata": {
            "active_prefix_checkpoint_count": round_,
            "checkpoint_depth": round_,
            "history_count": round_,
            "history_checkpoint_complete": True,
            "strict_replay_passed": True,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "estimator_call_ledger_checkpoint": {
                "S_alg": s_alg,
                "S_unique": s_unique,
                "checkpoint_depth": round_,
                "checkpoint_reason": "iteration_done",
                "current_round_finalized": True,
                "enabled": True,
                "ledger_fingerprint": ledger_fingerprint,
                "ledger_schema": "estimator_call_ledger_v1",
                "ledger_scope": "single_route",
                "path": ledger_path,
                "raw_occurrence_count": s_alg,
                "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
                "sha256": ledger_sha256,
                "status": "complete",
                "unique_primitive_count": s_unique,
            },
            "verified_singleton_resume_sidecar": {
                "enabled": True,
                "no_credentials_serialized": True,
                "path": sidecar_path,
                "schema": (
                    "static_adapt_verified_singleton_resume_sidecar_pointer_v1"
                ),
                "sha256": sidecar_sha256,
                "sidecar_schema": (
                    "static_adapt_signed_active_prefix_resume_sidecar_v2"
                ),
                "source_projection_schema": (
                    "static_adapt_verified_singleton_resume_source_projection_v1"
                ),
                "source_projection_sha256": source_projection_sha256,
                "status": "complete",
            },
        },
    }


CELL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "regime_id": "weak_strong",
        "resume_round": 56,
        "source_protocol_sha256": "8a5d5b62d1e008c6c3a428898156c38c9458f24035caa9d17159a771dcfb6b2e",
        "source_job_file_sha256": "9be77f0f3a9611ff2a1ff12f0573041796ac3edffff6653e6ff4ac91cad1c4f1",
        "visible_error": 3.840699455714969e-05,
        "source_s_alg": 360847,
        "source_kind": "local_recoverable_checkpoint_triplet",
        "source_members": _local_triplet(
            "output/local_runs/paper_i_page10_macro_then_singleton_phase23_qiskit_no_lanes_weak_strong_r50_to_r70_20260808_v2_repaired/checkpoints",
            ("7a6f13131db8acde941efe7e0212da5f1d843cfa21495be9e76b58c54b2defe4", 729622832),
            ("af44e396f2e2cc24ee61a1df15c8e56c1278dcd24f69daee306b19a7d093f15e", 1206441881),
            ("7894cbcf53bb2f4148debf1dcef28930195bbad94bb504502800397f69352bb0", 5322),
        ),
        "v1_checkpoint_validation": _v1_checkpoint_validation(
            regime_id="weak_strong",
            manifest_file_sha256=(
                "583d913c92896a0b6b070804110fb832a4ac2f86d96d538c76960ee318242372"
            ),
            manifest_sha256=(
                "f62323e53a4d3bc0de4edf7a4009d0cfa278863cae4ac37288fc0e7915caf682"
            ),
            archive_sha256=(
                "960c404f34d1424b0f41a1105a8df4124ea55df2a5334aeb0479f75025d4b6bf"
            ),
            archive_size_bytes=430224574,
            round_=56,
            ledger_sha256=(
                "af44e396f2e2cc24ee61a1df15c8e56c1278dcd24f69daee306b19a7d093f15e"
            ),
            ledger_path=(
                "current.estimator_call_ledger_checkpoint.af44e396f2e2cc24.json"
            ),
            ledger_fingerprint=(
                "0b969f01ca3b46a5f82a42eb752bd1bb2845d3eef5d27cef97f0f2cab667a8f8"
            ),
            s_alg=360847,
            s_unique=344459,
            sidecar_sha256=(
                "7894cbcf53bb2f4148debf1dcef28930195bbad94bb504502800397f69352bb0"
            ),
            sidecar_path=(
                "current.verified_singleton_resume.7894cbcf53bb2f41.json"
            ),
            source_projection_sha256=(
                "2f937394f2111aaced53fae7dd6dfd0f30bdb3c0bae82eba638f6bd38d7e1f43"
            ),
        ),
    },
    {
        "regime_id": "intermediate_strong",
        "resume_round": 51,
        "source_protocol_sha256": "0541200b8ce50ca3a8876157038e7308e64dd7c75c69f211134f70122e9cf8aa",
        "source_job_file_sha256": "949495e6120de6b90e7afa4e70d41e473e6680b0d5e5d2657b8fd58cf07cf468",
        "visible_error": 1.2290466894049334e-04,
        "source_s_alg": 339815,
        "source_kind": "local_recoverable_checkpoint_triplet",
        "source_members": _local_triplet(
            "output/local_runs/paper_i_page10_macro_then_singleton_phase23_qiskit_no_lanes_intermediate_strong_r50_to_r70_20260808_v3_repaired/checkpoints",
            ("278d47070d69c607e678a79ad7852249ada9235bffabe6524d30a3a05667d28d", 651933759),
            ("56946924dc38c4c1d05ac3718f0b9faf6f6701be87e703bdca9b30c39daa3fcc", 1107998560),
            ("54a0f615c809643e27874a7e1c1000227a5817f986157086692deb575f27bdeb", 5213),
        ),
        "v1_checkpoint_validation": _v1_checkpoint_validation(
            regime_id="intermediate_strong",
            manifest_file_sha256=(
                "c41011639e86f13d675754a8c5df70b07932818b4042da292e59d92148bd5dc9"
            ),
            manifest_sha256=(
                "eca3ed2d868d97819c834e04841c48b07c9514c6326d5f2efb7f53ed7724982a"
            ),
            archive_sha256=(
                "0bc03a2f4ce126a1ddd3996c1fb48a37513528bdcd42ad5aec7881242d980399"
            ),
            archive_size_bytes=393087213,
            round_=51,
            ledger_sha256=(
                "56946924dc38c4c1d05ac3718f0b9faf6f6701be87e703bdca9b30c39daa3fcc"
            ),
            ledger_path=(
                "current.estimator_call_ledger_checkpoint.56946924dc38c4c1.json"
            ),
            ledger_fingerprint=(
                "2c1691ac3a7f12dd7ea835f1276f392ac09cb0b86973820e75d747a398ccafea"
            ),
            s_alg=339815,
            s_unique=322713,
            sidecar_sha256=(
                "54a0f615c809643e27874a7e1c1000227a5817f986157086692deb575f27bdeb"
            ),
            sidecar_path=(
                "current.verified_singleton_resume.54a0f615c809643e.json"
            ),
            source_projection_sha256=(
                "69a9aa0e79ad7f8d64d348bb6a13ee05fc8aeeeb1295fa2331a377816b304806"
            ),
        ),
    },
    {
        "regime_id": "strong_strong_u8",
        "resume_round": 50,
        "source_protocol_sha256": "c12fc9afb3de555270349d0a73532952a2c8bd7d49f8100057c7d7dd357e2889",
        "source_job_file_sha256": "6a0ded3a6a5e20bf08422324168999d7c43024c6007857d0d99cff3a92676e43",
        "visible_error": 4.51466652950927e-09,
        "source_s_alg": 446799,
        "source_kind": "validated_chtc_attempt_archive",
        "source_archive": {
            "path": "chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260807_macro_then_singleton_phase123_qiskit_phase23_no_lanes_v1/9600705.5_strong_strong_u8.tar.gz",
            "sha256": "c2ca5029334be90d664e4798483de9b095fe4af024d44027ef5377632efe1d7b",
            "size_bytes": 737842102,
        },
        "source_worker_receipt": {
            "path": "chtc/paper_i_ra_adapt_repair_20260727/retrieved_chtc_20260807_macro_then_singleton_phase123_qiskit_phase23_no_lanes_v1/9600705.5_strong_strong_u8/worker_receipt.json",
            "sha256": "605cdd3972bc2be3e78b394c1814302c647d8886e0a103a832986d58950217d6",
            "canonical_sha256": "00664f7da88cc6a2c08c5451e318c2169c6c2d57c411dabf290f8257d61c88d7",
        },
        "source_members": (
            {"role": "checkpoint", "source_member": f"./runs/{source_execution_id('strong_strong_u8')}/checkpoints/current.json", "archive_path": "resume/current.json", "sha256": "d27e0fd70e39edff2b893355ab6e379e600ed63f18cf81d2459195b47210afb1", "size_bytes": 821342544},
            {"role": "estimator_ledger_checkpoint", "source_member": f"./runs/{source_execution_id('strong_strong_u8')}/checkpoints/current.estimator_call_ledger_checkpoint.2ba00ca7a0514b28.json", "archive_path": "resume/current.estimator_call_ledger_checkpoint.2ba00ca7a0514b28.json", "sha256": "2ba00ca7a0514b28c962178b8c7d4e5552c7e895e5281143e2fd4de649e3d953", "size_bytes": 1477226276},
            {"role": "verified_resume_sidecar", "source_member": f"./runs/{source_execution_id('strong_strong_u8')}/checkpoints/current.verified_singleton_resume.a157f7c35cef4334.json", "archive_path": "resume/current.verified_singleton_resume.a157f7c35cef4334.json", "sha256": "a157f7c35cef43340e1ef39a9e71fde9dcbc9d43ecb584ece0b746118a9bb8db", "size_bytes": 5183},
        ),
        "v1_checkpoint_validation": _v1_checkpoint_validation(
            regime_id="strong_strong_u8",
            manifest_file_sha256=(
                "fcdadf87f57a57f7c555522a2221543b2f05b6a80dd5fcbcd6428bad02b28a95"
            ),
            manifest_sha256=(
                "594a241d86b00e9b53be592320fd8998e93dd9970977af02ca580b4660650b77"
            ),
            archive_sha256=(
                "f40aa01924978922e810d70cc5619a54925115bbb6b78bf66d49839ca6a09534"
            ),
            archive_size_bytes=512379812,
            round_=50,
            ledger_sha256=(
                "2ba00ca7a0514b28c962178b8c7d4e5552c7e895e5281143e2fd4de649e3d953"
            ),
            ledger_path=(
                "current.estimator_call_ledger_checkpoint.2ba00ca7a0514b28.json"
            ),
            ledger_fingerprint=(
                "19d6f1dbbcc60d8c6cc68769dcafdd1bd4b4bd67ec9193c86e0b00e5e4cb5391"
            ),
            s_alg=446799,
            s_unique=434403,
            sidecar_sha256=(
                "a157f7c35cef43340e1ef39a9e71fde9dcbc9d43ecb584ece0b746118a9bb8db"
            ),
            sidecar_path=(
                "current.verified_singleton_resume.a157f7c35cef4334.json"
            ),
            source_projection_sha256=(
                "b1d3a49c29f2a4b5e5debba2fc2d6fa2d30a936248683d3f9aec2c525d90d1fb"
            ),
        ),
    },
)

RESOURCE_ENVELOPE = {
    "request_cpus": 4,
    "request_memory_mb": 32768,
    "request_disk_mb": 61440,
    "max_runtime_seconds": 259200,
    "basis": (
        "page10_observed_peak_memory_4883_7325_12208_mib_"
        "plus_checkpoint_hydration_headroom_v1"
    ),
}

PACKAGE_MANIFEST_SCHEMA = "paper_i_page10_strong_r70_continuation_package_v1"
JOB_SCHEMA = "paper_i_page10_strong_r70_continuation_job_v1"
BUNDLE_MANIFEST_SCHEMA = "paper_i_page10_strong_r70_continuation_bundle_v1"
ACTIVATION_SCHEMA = "paper_i_page10_strong_r70_continuation_activation_v1"
AUTHORIZATION_SCHEMA = "paper_i_page10_strong_r70_continuation_authorization_v1"

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
            source_validation = checkpoint_validation.get("source_validation")
            if (
                checkpoint_validation.get("schema")
                != "paper_i_page10_checkpoint_validation_receipt_v1"
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
                != "inherited_v1_full_stream_validation_exact_bytes_v1"
                or checkpoint_validation.get("worker_validation_scope")
                != "stream_authenticate_all_three_members_then_strict_resume_replay_v1"
                or not isinstance(source_validation, Mapping)
                or source_validation.get("archive") != archive
                or source_validation.get("source_package_manifest", {}).get(
                    "sha256"
                )
                != V1_CONTINUATION_MANIFEST_FILE_SHA256
                or source_validation.get("source_package_manifest", {}).get(
                    "canonical_sha256"
                )
                != V1_CONTINUATION_MANIFEST_SHA256
            ):
                raise PackageContractError(
                    "Inherited checkpoint validation authority drifted."
                )
            raw_metadata = checkpoint_validation.get("metadata")
            if not isinstance(raw_metadata, Mapping):
                raise PackageContractError(
                    "Checkpoint validation metadata is absent."
                )
            metadata = dict(raw_metadata)
            validation_source = (
                "inherited_v1_full_stream_validation_exact_bytes_v1"
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
    ):
        raise PackageContractError(
            f"Resume checkpoint is not the authenticated round-{expected_round} prefix."
        )
    return {
        "metadata": metadata,
        "members_by_role": by_role,
        "checkpoint_validation_source": validation_source,
    }
