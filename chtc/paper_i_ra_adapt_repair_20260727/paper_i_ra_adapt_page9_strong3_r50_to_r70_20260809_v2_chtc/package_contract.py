#!/usr/bin/env python3
"""Closed contract for the Page-9 strong-sector k=50 -> 70 continuations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping

from vendored_ijson_python import JSONError, parse as stream_json_parse


PACKAGE_ID = "paper_i_ra_adapt_page9_strong3_r50_to_r70_20260809_v2_chtc"
CAMPAIGN_ID = "paper_i_ra_adapt_page9_strong3_r50_to_r70_v2"
BUNDLE_ID = "paper_i_ra_adapt_page9_strong3_r70_continuation_v2"
RUN_CLASS = "candidate"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70

BASE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_phase3_qiskit_denominator_no_lanes_"
    "tau1em6_r50_20260807_v3_chtc"
)
BASE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "059c4d8f6a53c14aabfe99e7c52dd7279bf91cdec1687e8cc6b74529a5b21299"
)
BASE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "c327d23791cbc3deaabd74e5a7fcd0171edd5e50d8cc93d3a873258cdd61ade9"
)
BASE_RUNNER_SHA256 = (
    "f29924f456f97b2ec31b1e88244b7bfe4cbe65e6c3dfa80e0c217acf27c0a714"
)
BASE_SOURCE_ARCHIVE_SHA256 = (
    "6587cc07965580477e11a764e84e7c08868da1625f1b9d24e7e82f8fe89fca49"
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
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/"
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "phase3_qiskit_denominator_no_lanes_page9_adapter.json"
)
VISIBLE_ADAPTER_FILE_SHA256 = (
    "00e1d5a635d8d013ba8d7bdc196377eb270b4c929d20f836d5d8faa3b74ef85c"
)
VISIBLE_ADAPTER_CANONICAL_SHA256 = (
    "f2c3d1586c0b56f4d301fa26f122f77185d1d6af3d94fae7bb1f5c1e004b99a8"
)
VISIBLE_ADAPTER_SCHEMA = "paper_i_phase3_qiskit_no_lanes_page9_adapter_v1"
VISIBLE_PAGE_ID = "phase3_qiskit_denominator_no_lanes_singleton_r50_partial_v1"

ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "qiskit_full_ansatz_positive_marginal_denominator_phase3_only_"
    "no_lanes_tau1em6_v1"
)
ROUTE_CONTRACT_SHA256 = (
    "e649eaa50428f6f396c4ab6cf25542a21add58115beb61d42df32408ad1399b6"
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_plateau_commutation_"
    "qiskit_phase3_denominator_no_lanes_tau1em6_v1"
)
ROUTE_ID = "ra_global_singleton_plateau_commutation"
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_global_single_pauli_word_candidate_adapter_v1"
)
REMOTE_IMAGE_RELATIVE = Path("chtc/phase3_optuna/image.sif")
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
STAGING_OUTPUT_ROOT = Path(
    "/staging/jsstrobel/paper_i_ra_adapt_page9_strong3_r70_20260809_v2/attempts"
)

RESOURCE_ENVELOPE = {
    "request_cpus": 4,
    "request_memory_mb": 262_144,
    "request_disk_mb": 102_400,
    "max_runtime_seconds": 259_200,
    "basis": "page9_nph7_observed_high_memory_plus_resume_hydration_v1",
}

REGIMES = ("weak_strong", "intermediate_strong", "strong_strong_u8")
SOURCE_EXECUTION_IDS = {
    regime: (
        f"phase3_qiskit_denominator_no_lanes__{regime}__nph7__"
        "ra_global_singleton_plateau_commutation"
    )
    for regime in REGIMES
}

PACKAGE_SCHEMA = "paper_i_page9_strong3_r70_continuation_package_v2"
JOB_SCHEMA = "paper_i_page9_strong3_r70_continuation_job_v2"
RESUME_MATERIALIZATION_SCHEMA = (
    "paper_i_page9_strong3_r70_resume_materialization_v2"
)
AUTHORIZATION_SCHEMA = "paper_i_page9_strong3_r70_execution_authorization_v2"
ACTIVATION_REQUEST_SCHEMA = "paper_i_page9_strong3_r70_activation_request_v2"
ACTIVATION_MANIFEST_SCHEMA = "paper_i_page9_strong3_r70_activation_manifest_v2"

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "materialize_resume_input.py",
    "run_cell.py",
    "activate_package.py",
    "validate_package.py",
    "execute_authorized_job.sh",
    "submit.sub.in",
    "README.md",
    "vendored_ijson_python.py",
    "IJSON_LICENSE.txt",
)
GENERATED_PATHS = (
    "bundle_manifest.json",
    "protocols",
    "jobs",
    "prefix_anchors",
    "source_overlay",
    "source_composition.json",
    "visible_source_map.json",
    "resolver_traces",
    "queue.tsv",
    "execution_plan.json",
    "source_lock_audit.json",
    "package_manifest.json",
)


class PackageContractError(RuntimeError):
    """Fail-closed package, materialization, or worker violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    expected = unsigned.pop("sha256", None)
    observed = canonical_sha256(unsigned)
    if expected != observed:
        raise PackageContractError(f"{label} canonical digest drifted.")
    return observed


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
        raise PackageContractError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a nonempty path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise PackageContractError(f"Unsafe {label}: {value!r}.")
    return Path(*pure.parts)


def safe_absolute_posix_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PackageContractError(f"{label} must be a nonempty path.")
    pure = PurePosixPath(value)
    if not pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise PackageContractError(f"Unsafe {label}: {value!r}.")
    if not str(pure).startswith("/staging/jsstrobel/"):
        raise PackageContractError(f"{label} must be under /staging/jsstrobel.")
    return Path(str(pure))


def repo_root_from_script(path: str | Path) -> Path:
    current = Path(path).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines/static_adapt"
        ).is_dir():
            return candidate
    raise PackageContractError("Active repository root was not found.")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PackageContractError(f"Unsafe bound file: {path}")
    try:
        relative = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(f"Bound file escaped root: {path}") from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=path.name)
    return {
        **file_binding(path, relative_to=relative_to),
        "canonical_sha256": verify_self_digest(payload, label=path.name),
    }


def source_execution_id(regime: str) -> str:
    try:
        return SOURCE_EXECUTION_IDS[regime]
    except KeyError as exc:
        raise PackageContractError(f"Unsupported regime: {regime}") from exc


def continuation_execution_id(regime: str) -> str:
    return f"{source_execution_id(regime)}__resume_r50_to_r70"


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(continuation_execution_id(regime) for regime in REGIMES)


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
                    scalar_differences(before[key], after[key], path=(*path, key))
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


def prefix_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    trace = summary.get("accepted_error_trace")
    requested = summary.get("requested_rounds")
    if not isinstance(trace, list) or not isinstance(requested, list):
        raise PackageContractError("Paper-I summary lacks a prefix trajectory.")
    prefix_trace = [
        dict(row)
        for row in trace
        if isinstance(row, Mapping)
        and int(row.get("controller_round", -1)) <= SOURCE_HORIZON
    ]
    terminal_rows = [
        dict(row)
        for row in requested
        if isinstance(row, Mapping)
        and int(row.get("controller_round", -1)) == SOURCE_HORIZON
    ]
    if (
        len(prefix_trace) != SOURCE_HORIZON
        or [int(row["controller_round"]) for row in prefix_trace]
        != list(range(1, SOURCE_HORIZON + 1))
        or len(terminal_rows) != 1
    ):
        raise PackageContractError("Round-50 summary prefix is incomplete.")
    terminal = terminal_rows[0]
    keep = {
        key: terminal[key]
        for key in (
            "absolute_energy_error",
            "active_ansatz_depth",
            "algorithmic_work",
            "controller_round",
            "prefix",
        )
        if key in terminal
    }
    if set(keep) != {
        "absolute_energy_error",
        "active_ansatz_depth",
        "algorithmic_work",
        "controller_round",
        "prefix",
    }:
        raise PackageContractError("Round-50 requested prefix is incomplete.")
    return digested(
        {
            "schema": "paper_i_page9_r50_prefix_projection_v1",
            "source_horizon": SOURCE_HORIZON,
            "accepted_error_trace": prefix_trace,
            "requested_round": keep,
        }
    )


def _checkpoint_metadata(stream: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {"active_prefix_checkpoint_count": 0}
    ledger: dict[str, Any] = {}
    resume: dict[str, Any] = {}
    for prefix, event, value in stream_json_parse(stream):
        if (
            prefix == "adapt_vqe.active_prefix_checkpoints.item"
            and event == "start_map"
        ):
            metadata["active_prefix_checkpoint_count"] += 1
        elif prefix == "checkpoint.depth" and event in {"integer", "number"}:
            metadata["checkpoint_depth"] = int(value)
        elif prefix == "adapt_vqe.history_count" and event in {"integer", "number"}:
            metadata["history_count"] = int(value)
        elif (
            prefix == "adapt_vqe.history_checkpoint_complete"
            and event == "boolean"
        ):
            metadata["history_checkpoint_complete"] = value
        elif prefix == "adapt_vqe.strict_replay.passed" and event == "boolean":
            metadata["strict_replay_passed"] = value
        elif (
            prefix == "adapt_vqe.sr_route_profile_contract_sha256"
            and event == "string"
        ):
            metadata["route_contract_sha256"] = value
        elif prefix.startswith(
            "adapt_vqe.estimator_call_ledger_checkpoint."
        ) and event in {"boolean", "integer", "number", "string"}:
            ledger[prefix.rsplit(".", 1)[-1]] = value
        elif prefix.startswith(
            "adapt_vqe.verified_singleton_resume_sidecar."
        ) and event in {"boolean", "integer", "number", "string"}:
            resume[prefix.rsplit(".", 1)[-1]] = value
    metadata["estimator_call_ledger_checkpoint"] = ledger
    metadata["verified_singleton_resume_sidecar"] = resume
    return metadata


def validate_resume_archive(
    archive_path: Path,
    manifest: Mapping[str, Any],
    *,
    expected_round: int = SOURCE_HORIZON,
) -> dict[str, Any]:
    """Authenticate a content-addressed, pointer-closed checkpoint triplet."""

    verify_self_digest(manifest, label="resume materialization")
    archive = manifest.get("archive")
    raw_members = manifest.get("members")
    if not isinstance(archive, Mapping) or not isinstance(raw_members, list):
        raise PackageContractError("Resume materialization is incomplete.")
    if (
        manifest.get("schema") != RESUME_MATERIALIZATION_SCHEMA
        or manifest.get("status") != "passed_pointer_closed_triplet"
        or manifest.get("resume_round") != expected_round
        or manifest.get("member_count") != 3
        or manifest.get("pointer_closed") is not True
        or not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size != int(archive.get("size_bytes", -1))
        or sha256_file(archive_path) != archive.get("sha256")
    ):
        raise PackageContractError("Resume archive binding drifted.")
    by_path: dict[str, Mapping[str, Any]] = {}
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw in raw_members:
        if not isinstance(raw, Mapping):
            raise PackageContractError("Resume member is malformed.")
        relative = safe_relative_path(
            raw.get("path"), label="resume member"
        ).as_posix()
        role = str(raw.get("role", ""))
        if relative in by_path or role in by_role:
            raise PackageContractError("Resume member identity is duplicated.")
        by_path[relative] = raw
        by_role[role] = raw
    if set(by_role) != {
        "checkpoint",
        "estimator_ledger_checkpoint",
        "verified_resume_sidecar",
    }:
        raise PackageContractError("Resume member roles are not closed.")
    observed: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as opened:
            for member in opened:
                relative = safe_relative_path(
                    member.name, label="resume tar member"
                ).as_posix()
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
                    raise PackageContractError(
                        f"Unreadable resume member: {relative}"
                    )
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
        with tarfile.open(archive_path, "r:gz") as opened:
            checkpoint_member = opened.getmember(str(checkpoint_row["path"]))
            checkpoint_stream = opened.extractfile(checkpoint_member)
            if checkpoint_stream is None:
                raise PackageContractError("Resume checkpoint is unreadable.")
            metadata = _checkpoint_metadata(checkpoint_stream)
    except (OSError, EOFError, tarfile.TarError, JSONError) as exc:
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
            or (
                pointer.get("size_bytes") is not None
                and int(pointer["size_bytes"])
                != int(row.get("size_bytes", -1))
            )
            or (
                role == "verified_resume_sidecar"
                and pointer.get("enabled") is not True
            )
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
    return {"metadata": metadata, "members_by_role": by_role}
