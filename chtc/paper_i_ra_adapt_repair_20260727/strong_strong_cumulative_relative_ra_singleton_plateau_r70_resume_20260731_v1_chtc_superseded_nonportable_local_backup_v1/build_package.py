#!/usr/bin/env python3
"""Build the sealed one-row cumulative-relative r50->r70 package."""

from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any, BinaryIO, Mapping

import ijson


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ALLOWED_OPERATIONAL_SOURCE_DELTAS,
    BASE_IMPLEMENTATION_INVENTORY_SHA256,
    CAMPAIGN_ID,
    CANDIDATE_REPRESENTATION,
    DERIVED_PROTOCOL_CHANGED_PATHS,
    EXECUTION_ID,
    KNOWN_JSON_BINDINGS,
    MATERIALIZATION_ROOT,
    PACKAGE_ID,
    PLATEAU_COMPARISON,
    PLATEAU_RATIO_THRESHOLD,
    PLATEAU_TRIGGER,
    R50_CHECKPOINT,
    R50_CHECKPOINT_BINDING,
    R50_CHECKPOINT_ROOT,
    R50_LEDGER_BINDING,
    R50_LEDGER_SIDECAR,
    R50_RESULT_BINDING,
    R50_ROOT,
    R50_SUMMARY_BINDING,
    R50_VERIFIED_BINDING,
    R50_VERIFIED_SIDECAR,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_CONTRACT_SHA256,
    SOURCE_BUNDLE_ROOT,
    SOURCE_CELL_ID,
    SOURCE_HORIZON,
    SOURCE_PROTOCOL,
    SOURCE_PROTOCOL_CANONICAL_SHA256,
    SOURCE_PROTOCOL_FILE_SHA256,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    scalar_differences,
    sha256_file,
    verify_self_digest,
)


CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
    "README.md",
)
GENERATED_PATHS = (
    "derived_protocols",
    "lineage",
    "resume_inputs",
    "source",
    "execution_plan.json",
    "source_lock_audit.json",
    "resume_input.json",
    "job.json",
    "package_manifest.json",
)
SOURCE_RUNNER = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_cumulative_plateau_pair_20260731.py"
)


def _exclusive_write(
    path: Path, data: bytes, *, executable: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite {path}.")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _exclusive_write(path, canonical_json_bytes(value) + b"\n")


def _exclusive_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PackageContractError(f"Refusing to overwrite {destination}.")
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as input_stream:
            with temporary.open("xb") as output_stream:
                shutil.copyfileobj(
                    input_stream, output_stream, length=1024 * 1024
                )
                output_stream.flush()
                os.fsync(output_stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _package_binding(
    path: Path, *, canonical: bool = False
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": path.relative_to(PACKAGE_DIR).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        payload = load_json(path)
        binding["canonical_sha256"] = verify_self_digest(
            payload, label=path.name
        )
    return binding


def _validate_known_json(repo_root: Path) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for role, (relative, expected_file, expected_canonical) in sorted(
        KNOWN_JSON_BINDINGS.items()
    ):
        path = repo_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_file
        ):
            raise PackageContractError(f"{role} byte binding drifted.")
        payload = load_json(path, label=role)
        if verify_self_digest(payload, label=role) != expected_canonical:
            raise PackageContractError(f"{role} canonical binding drifted.")
        validated[role] = payload
    return validated


def _validate_r50_lineage(
    repo_root: Path, known: Mapping[str, Mapping[str, Any]]
) -> None:
    r20_terminal = known["r20_terminal"]
    r30_terminal = known["r30_terminal"]
    r30_repair = known["r30_repair"]
    terminal = known["r50_terminal"]
    repair = known["r50_repair"]
    manifest = known["r50_manifest"]
    authorization = known["r50_authorization"]
    protocol = known["source_protocol"]
    r20_terminal_sha256 = r20_terminal.get("sha256")
    r30_terminal_sha256 = r30_terminal.get("sha256")
    r30_checkpoint = r30_terminal.get("canonical_resume_checkpoint")
    r30_sidecars = r30_terminal.get(
        "canonical_resume_checkpoint_sidecars"
    )
    route = protocol.get("route_contract")
    invariants = (
        route.get("semantic_invariants")
        if isinstance(route, Mapping)
        else None
    )
    if (
        not isinstance(r30_checkpoint, Mapping)
        or not isinstance(r30_sidecars, Mapping)
        or r30_terminal.get("source_round_20_terminal_receipt_sha256")
        != r20_terminal_sha256
        or r30_repair.get("corrected_checkpoint") != r30_checkpoint
        or r30_repair.get("corrected_checkpoint_sidecars")
        != r30_sidecars
        or manifest.get("source_terminal_receipt_sha256")
        != r30_terminal_sha256
        or authorization.get("source_terminal_receipt_sha256")
        != r30_terminal_sha256
        or terminal.get("source_terminal_receipt_sha256")
        != r30_terminal_sha256
        or manifest.get("resume_input") != r30_checkpoint
        or manifest.get("resume_input_sidecars") != r30_sidecars
        or authorization.get("source_checkpoint_sha256")
        != r30_checkpoint.get("sha256")
        or terminal.get("source_checkpoint_sha256")
        != r30_checkpoint.get("sha256")
        or terminal.get("status") != "passed"
        or terminal.get("accepted_controller_rounds") != SOURCE_HORIZON
        or terminal.get("protocol_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or terminal.get("canonical_resume_checkpoint")
        != R50_CHECKPOINT_BINDING
        or terminal.get("canonical_resume_checkpoint_sidecars")
        != {
            "estimator_ledger_checkpoint": R50_LEDGER_BINDING,
            "verified_singleton_resume": R50_VERIFIED_BINDING,
        }
        or terminal.get("canonical_resume_repair_receipt_sha256")
        != repair.get("sha256")
        or repair.get("status") != "passed"
        or repair.get("scientific_state_changed") is not False
        or repair.get("canonical_resume_validation") != "passed"
        or repair.get("controller_round") != SOURCE_HORIZON
        or repair.get("corrected_checkpoint") != R50_CHECKPOINT_BINDING
        or repair.get("corrected_checkpoint_sidecars")
        != {
            "estimator_ledger_checkpoint": R50_LEDGER_BINDING,
            "verified_singleton_resume": R50_VERIFIED_BINDING,
        }
        or manifest.get("protocol_sha256")
        != SOURCE_PROTOCOL_CANONICAL_SHA256
        or manifest.get("target_round") != SOURCE_HORIZON
        or not isinstance(invariants, Mapping)
        or route.get("sha256") != ROUTE_CONTRACT_SHA256
        or invariants.get("plateau_cumulative_decrease_ratio_threshold")
        != PLATEAU_RATIO_THRESHOLD
        or invariants.get("plateau_threshold_comparison")
        != PLATEAU_COMPARISON
        or invariants.get("plateau_trigger_source") != PLATEAU_TRIGGER
    ):
        raise PackageContractError(
            "Corrected cumulative-relative r50 lineage drifted."
        )
    for path, binding, label in (
        (R50_CHECKPOINT, R50_CHECKPOINT_BINDING, "r50 checkpoint"),
        (R50_LEDGER_SIDECAR, R50_LEDGER_BINDING, "r50 ledger"),
        (R50_VERIFIED_SIDECAR, R50_VERIFIED_BINDING, "r50 resume sidecar"),
    ):
        absolute = repo_root / path
        if (
            not absolute.is_file()
            or absolute.is_symlink()
            or absolute.stat().st_size != binding["size_bytes"]
            or sha256_file(absolute) != binding["sha256"]
        ):
            raise PackageContractError(f"{label} binding drifted.")
    for binding, label in (
        (R50_RESULT_BINDING, "r50 result"),
        (R50_SUMMARY_BINDING, "r50 summary"),
    ):
        absolute = repo_root / binding["path"]
        if (
            not absolute.is_file()
            or absolute.is_symlink()
            or absolute.stat().st_size != binding["size_bytes"]
            or sha256_file(absolute) != binding["sha256"]
        ):
            raise PackageContractError(f"{label} binding drifted.")


def _derive_protocol(
    source_payload: Mapping[str, Any]
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        resolved_ra_adapt_protocol_from_mapping,
    )

    derived = copy.deepcopy(dict(source_payload))
    derived["horizon"] = TARGET_HORIZON
    derived["request"]["execution"]["stop"][
        "maximum_controller_rounds"
    ] = TARGET_HORIZON
    derived["stopping_rule"]["maximum_controller_rounds"] = TARGET_HORIZON
    derived.pop("sha256", None)
    derived["sha256"] = canonical_sha256(derived)
    typed = resolved_ra_adapt_protocol_from_mapping(derived)
    if typed.sha256 != derived["sha256"]:
        raise PackageContractError("Derived protocol did not rehydrate.")
    changed = sorted(
        ".".join(str(component) for component in path)
        for path, _before, _after in scalar_differences(
            source_payload, derived
        )
    )
    if (
        changed != list(DERIVED_PROTOCOL_CHANGED_PATHS)
        or typed.active_gradient_policy != ACTIVE_GRADIENT_POLICY
        or typed.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
        or typed.candidate_representation != CANDIDATE_REPRESENTATION
        or typed.route_contract != source_payload.get("route_contract")
        or typed.problem.to_dict() != source_payload.get("problem")
        or typed.bundle_materialization.to_dict()
        != source_payload.get("bundle_materialization")
    ):
        raise PackageContractError(
            f"Derived protocol changed non-horizon settings: {changed}"
        )
    return derived


def _source_members(
    *, repo_root: Path, source_locks: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    implementation = source_locks.get("implementation_sources")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("sha256")
        != BASE_IMPLEMENTATION_INVENTORY_SHA256
        or not isinstance(implementation.get("files"), list)
    ):
        raise PackageContractError("Implementation inventory drifted.")
    declared: dict[str, str] = {}
    paths: set[str] = set()
    changed: list[dict[str, Any]] = []
    for raw in implementation["files"]:
        if not isinstance(raw, Mapping):
            raise PackageContractError("Malformed implementation row.")
        relative = safe_relative_path(
            raw.get("path"), label="implementation source"
        ).as_posix()
        prior = str(raw.get("sha256", ""))
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise PackageContractError(f"Missing source member: {relative}")
        observed = sha256_file(source)
        if observed != prior:
            reason = ALLOWED_OPERATIONAL_SOURCE_DELTAS.get(relative)
            if reason is None:
                raise PackageContractError(
                    f"Unapproved implementation drift: {relative}"
                )
            changed.append(
                {
                    "path": relative,
                    "base_sha256": prior,
                    "effective_sha256": observed,
                    "effective_size_bytes": source.stat().st_size,
                    "classification": "operational_only",
                    "reason": reason,
                    "scientific_protocol_change": False,
                    "controller_semantics_change": False,
                    "accepted_state_change": False,
                }
            )
        declared[relative] = observed
        paths.add(relative)
    if {row["path"] for row in changed} != set(
        ALLOWED_OPERATIONAL_SOURCE_DELTAS
    ):
        raise PackageContractError(
            "Expected exactly the two reviewed operational source deltas."
        )

    global_sources = source_locks.get("global_sources")
    if not isinstance(global_sources, Mapping):
        raise PackageContractError("Global source locks are absent.")
    for raw in global_sources.values():
        if not isinstance(raw, Mapping):
            raise PackageContractError("Malformed global source lock.")
        relative = safe_relative_path(
            raw.get("path"), label="global source"
        ).as_posix()
        declared[relative] = str(raw.get("sha256", ""))
        paths.add(relative)

    for root in (MATERIALIZATION_ROOT,):
        for path in sorted((repo_root / root).rglob("*")):
            if path.is_file() and not path.is_symlink():
                paths.add(path.relative_to(repo_root).as_posix())
    paths.update(
        {
            (SOURCE_BUNDLE_ROOT / "bundle_manifest.json").as_posix(),
            (SOURCE_BUNDLE_ROOT / "source_locks.json").as_posix(),
            SOURCE_RUNNER.as_posix(),
            "requirements.txt",
        }
    )
    members: list[dict[str, Any]] = []
    for relative in sorted(paths):
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise PackageContractError(f"Unsafe source member: {relative}")
        observed = sha256_file(source)
        if relative in declared and observed != declared[relative]:
            raise PackageContractError(
                f"Declared non-operational source drift: {relative}"
            )
        members.append(
            {
                "path": relative,
                "sha256": observed,
                "size_bytes": source.stat().st_size,
            }
        )
    delta = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "effective_source_delta_v1"
            ),
            "status": "passed_operational_only",
            "base_implementation_inventory_sha256": (
                BASE_IMPLEMENTATION_INVENTORY_SHA256
            ),
            "changed_members": changed,
            "changed_member_count": len(changed),
            "scientific_settings_changed": [],
            "route_contract_changed": False,
            "protocol_changed_by_source_delta": False,
            "source_delta_authority": (
                "reviewed_estimator_ledger_and_occurrence_stable_"
                "writer_plumbing_repairs_2026_07_31"
            ),
        }
    )
    return members, delta


def _tar_info(path: str, size: int, *, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(path)
    info.size = size
    info.mode = 0o755 if executable else 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _write_source_archive(
    *, repo_root: Path, destination: Path, members: list[dict[str, Any]]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT
                ) as archive:
                    for row in members:
                        relative = str(row["path"])
                        source = repo_root / relative
                        if (
                            sha256_file(source) != row["sha256"]
                            or source.stat().st_size != row["size_bytes"]
                        ):
                            raise PackageContractError(
                                f"Source member changed during build: {relative}"
                            )
                        with source.open("rb") as stream:
                            archive.addfile(
                                _tar_info(
                                    relative,
                                    source.stat().st_size,
                                    executable=bool(
                                        source.stat().st_mode & 0o111
                                    ),
                                ),
                                stream,
                            )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"active_prefix_checkpoint_count": 0}
    ledger: dict[str, Any] = {}
    verified: dict[str, Any] = {}
    scalar = {"boolean", "integer", "double", "number", "null", "string"}
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if (
                prefix == "adapt_vqe.active_prefix_checkpoints.item"
                and event == "start_map"
            ):
                metadata["active_prefix_checkpoint_count"] += 1
            elif prefix == "schema_version" and event in scalar:
                metadata["schema_version"] = value
            elif prefix == "checkpoint.depth" and event in scalar:
                metadata["checkpoint_depth"] = int(value)
            elif prefix == "adapt_vqe.history_count" and event in scalar:
                metadata["history_count"] = int(value)
            elif (
                prefix == "adapt_vqe.history_checkpoint_complete"
                and event in scalar
            ):
                metadata["history_checkpoint_complete"] = bool(value)
            elif (
                prefix == "adapt_vqe.strict_replay.passed" and event in scalar
            ):
                metadata["strict_replay_passed"] = bool(value)
            elif prefix == "adapt_vqe.route_profile" and event in scalar:
                metadata["route_profile"] = str(value)
            elif (
                prefix
                == "adapt_vqe.sr_route_profile_contract_sha256"
                and event in scalar
            ):
                metadata["route_contract_sha256"] = str(value)
            elif (
                prefix.startswith(
                    "adapt_vqe.estimator_call_ledger_checkpoint."
                )
                and event in scalar
            ):
                ledger[prefix.rsplit(".", 1)[-1]] = value
            elif (
                prefix.startswith(
                    "adapt_vqe.verified_singleton_resume_sidecar."
                )
                and event in scalar
            ):
                verified[prefix.rsplit(".", 1)[-1]] = value
    metadata["estimator_call_ledger_checkpoint"] = ledger
    metadata["verified_singleton_resume_sidecar"] = verified
    return metadata


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        import hashlib

        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        self.digest.update(block)
        self.size += len(block)
        return block


def _write_resume_archive(
    *, repo_root: Path, destination: Path
) -> dict[str, Any]:
    checkpoint = repo_root / R50_CHECKPOINT
    metadata = _checkpoint_metadata(checkpoint)
    ledger_pointer = metadata["estimator_call_ledger_checkpoint"]
    verified_pointer = metadata["verified_singleton_resume_sidecar"]
    route_profile = load_json(repo_root / SOURCE_PROTOCOL)["route_contract"][
        "route_profile"
    ]
    if (
        metadata.get("schema_version")
        != "static_adapt_current_checkpoint_v1"
        or metadata.get("checkpoint_depth") != SOURCE_HORIZON
        or metadata.get("history_count") != SOURCE_HORIZON
        or metadata.get("active_prefix_checkpoint_count") != SOURCE_HORIZON
        or metadata.get("history_checkpoint_complete") is not True
        or metadata.get("strict_replay_passed") is not True
        or metadata.get("route_profile") != route_profile
        or metadata.get("route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or ledger_pointer.get("path") != R50_LEDGER_BINDING["path"]
        or ledger_pointer.get("sha256") != R50_LEDGER_BINDING["sha256"]
        or ledger_pointer.get("status") != "complete"
        or verified_pointer.get("path") != R50_VERIFIED_BINDING["path"]
        or verified_pointer.get("sha256")
        != R50_VERIFIED_BINDING["sha256"]
        or verified_pointer.get("status") != "complete"
        or verified_pointer.get("enabled") is not True
    ):
        raise PackageContractError("Canonical r50 pointer closure drifted.")
    sources = (
        ("checkpoint", checkpoint, R50_CHECKPOINT_BINDING),
        (
            "estimator_ledger_checkpoint",
            repo_root / R50_LEDGER_SIDECAR,
            R50_LEDGER_BINDING,
        ),
        (
            "verified_singleton_resume",
            repo_root / R50_VERIFIED_SIDECAR,
            R50_VERIFIED_BINDING,
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    members: list[dict[str, Any]] = []
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT
                ) as archive:
                    for role, source, expected in sources:
                        if (
                            source.stat().st_size != expected["size_bytes"]
                            or sha256_file(source) != expected["sha256"]
                        ):
                            raise PackageContractError(
                                f"Resume source drifted: {role}"
                            )
                        name = f"checkpoint/{source.name}"
                        with source.open("rb") as stream:
                            archive.addfile(
                                _tar_info(name, source.stat().st_size), stream
                            )
                        members.append(
                            {
                                "role": role,
                                "path": name,
                                "sha256": expected["sha256"],
                                "size_bytes": expected["size_bytes"],
                            }
                        )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "resume_input_v1"
            ),
            "status": "passed",
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "archive": _package_binding(destination),
            "checkpoint_path": f"checkpoint/{checkpoint.name}",
            "checkpoint_sha256": R50_CHECKPOINT_BINDING["sha256"],
            "member_count": len(members),
            "members": members,
            "pointer_closed": True,
            "authentication": metadata,
        }
    )


def _copy_lineage(repo_root: Path) -> dict[str, dict[str, Any]]:
    lineage_root = PACKAGE_DIR / "lineage"
    lineage_root.mkdir()
    bindings: dict[str, dict[str, Any]] = {}
    for role in (
        "r20_terminal",
        "r20_manifest",
        "r20_authorization",
        "r30_terminal",
        "r30_repair",
        "r50_manifest",
        "r50_authorization",
        "r50_terminal",
        "r50_repair",
    ):
        source, expected_file, expected_canonical = KNOWN_JSON_BINDINGS[role]
        destination = lineage_root / f"{role}.json"
        _exclusive_copy(repo_root / source, destination)
        binding = _package_binding(destination, canonical=True)
        if (
            binding["sha256"] != expected_file
            or binding["canonical_sha256"] != expected_canonical
        ):
            raise PackageContractError(f"Copied lineage drifted: {role}")
        bindings[role] = binding
    for role, source in (
        ("r50_result", repo_root / R50_RESULT_BINDING["path"]),
        ("r50_summary", repo_root / R50_SUMMARY_BINDING["path"]),
    ):
        destination = lineage_root / f"{role}.json"
        _exclusive_copy(source, destination)
        bindings[role] = _package_binding(destination)
    return bindings


def main() -> int:
    repo_root = repo_root_from_script(__file__)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    expected_relative = (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "strong_strong_cumulative_relative_ra_singleton_plateau_r70_"
        "resume_20260731_v1_chtc"
    )
    if PACKAGE_DIR.relative_to(repo_root).as_posix() != expected_relative:
        raise PackageContractError("Package root drifted.")
    forbidden_bytecode = [
        path
        for path in PACKAGE_DIR.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    if forbidden_bytecode:
        raise PackageContractError(
            "Unbound Python bytecode is forbidden in the transferred package."
        )
    for name in CONTROL_FILES:
        path = PACKAGE_DIR / name
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(f"Missing control file: {name}")
    for name in GENERATED_PATHS:
        path = PACKAGE_DIR / name
        if path.exists() or path.is_symlink():
            raise PackageContractError(f"Refusing to overwrite: {path}")

    known = _validate_known_json(repo_root)
    _validate_r50_lineage(repo_root, known)
    derived = _derive_protocol(known["source_protocol"])
    derived_path = (
        PACKAGE_DIR
        / "derived_protocols"
        / f"{EXECUTION_ID}.json"
    )
    _write_json(derived_path, derived)

    source_locks = known["materialization_source_locks"]
    source_members, source_delta = _source_members(
        repo_root=repo_root, source_locks=source_locks
    )
    source_root = PACKAGE_DIR / "source"
    source_root.mkdir()
    source_delta_path = source_root / "source_delta_receipt.json"
    _write_json(source_delta_path, source_delta)
    source_archive = source_root / "source_locked.tar.gz"
    _write_source_archive(
        repo_root=repo_root,
        destination=source_archive,
        members=source_members,
    )
    source_manifest = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "source_archive_manifest_v1"
            ),
            "status": "passed",
            "base_implementation_inventory_sha256": (
                BASE_IMPLEMENTATION_INVENTORY_SHA256
            ),
            "operational_source_delta_sha256": source_delta["sha256"],
            "archive": _package_binding(source_archive),
            "member_count": len(source_members),
            "members": source_members,
            "no_ambient_repo_imports": True,
        }
    )
    source_manifest_path = source_root / "source_archive_manifest.json"
    _write_json(source_manifest_path, source_manifest)

    resume_archive = (
        PACKAGE_DIR / "resume_inputs" / f"{EXECUTION_ID}.tar.gz"
    )
    resume_input = _write_resume_archive(
        repo_root=repo_root, destination=resume_archive
    )
    resume_input_path = PACKAGE_DIR / "resume_input.json"
    _write_json(resume_input_path, resume_input)
    lineage = _copy_lineage(repo_root)

    audit = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "source_lock_audit_v1"
            ),
            "status": "passed",
            "source_protocol": {
                "path": SOURCE_PROTOCOL.as_posix(),
                "sha256": SOURCE_PROTOCOL_FILE_SHA256,
                "canonical_sha256": SOURCE_PROTOCOL_CANONICAL_SHA256,
            },
            "derived_protocol": _package_binding(
                derived_path, canonical=True
            ),
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
            "changed_protocol_paths": list(
                DERIVED_PROTOCOL_CHANGED_PATHS
            ),
            "non_swept_settings_diff": [],
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "candidate_representation": CANDIDATE_REPRESENTATION,
            "plateau_cumulative_decrease_ratio_threshold": (
                PLATEAU_RATIO_THRESHOLD
            ),
            "plateau_threshold_comparison": PLATEAU_COMPARISON,
            "plateau_trigger_source": PLATEAU_TRIGGER,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "operational_source_delta": _package_binding(
                source_delta_path, canonical=True
            ),
            "scientific_settings_changed_by_source_delta": [],
            "resume_input_sha256": resume_input["sha256"],
            "r50_terminal_sha256": known["r50_terminal"]["sha256"],
            "r50_repair_sha256": known["r50_repair"]["sha256"],
        }
    )
    audit_path = PACKAGE_DIR / "source_lock_audit.json"
    _write_json(audit_path, audit)

    plan = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "execution_plan_v1"
            ),
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": "diagnostic_continuation",
            "execution_target": "chtc",
            "execution_id": EXECUTION_ID,
            "row_count": 1,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
            "source_archive_manifest_sha256": source_manifest["sha256"],
            "source_delta_receipt_sha256": source_delta["sha256"],
            "resume_input_sha256": resume_input["sha256"],
            "source_lock_audit_sha256": audit["sha256"],
            "lineage": lineage,
            "resources": {
                "request_cpus": 4,
                "request_memory_mb": 90_112,
                "request_disk_mb": 98_304,
                "max_runtime_seconds": 259_200,
            },
            "obsolete_predecessor_submission": {
                "cluster_id": 9_398_782,
                "proc_id": 0,
                "scientific_identity": "absolute_drop_plateau_obsolete",
                "external_state_revalidation_required": True,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
        }
    )
    plan_path = PACKAGE_DIR / "execution_plan.json"
    _write_json(plan_path, plan)

    job = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_job_v1"
            ),
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "source_cell_id": SOURCE_CELL_ID,
            "run_class": "diagnostic_continuation",
            "execution_mode": "authenticated_resume_50_to_70",
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "source_protocol_sha256": SOURCE_PROTOCOL_CANONICAL_SHA256,
            "derived_protocol": _package_binding(
                derived_path, canonical=True
            ),
            "source_archive": _package_binding(source_archive),
            "source_archive_manifest": _package_binding(
                source_manifest_path, canonical=True
            ),
            "source_delta_receipt": _package_binding(
                source_delta_path, canonical=True
            ),
            "resume_input": {
                key: resume_input[key]
                for key in (
                    "archive",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "member_count",
                    "members",
                    "pointer_closed",
                )
            },
            "lineage": lineage,
            "source_lock_audit_sha256": audit["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "candidate_representation": CANDIDATE_REPRESENTATION,
            "insertion_policy": "plateau_commutation",
            "plateau_cumulative_decrease_ratio_threshold": (
                PLATEAU_RATIO_THRESHOLD
            ),
            "plateau_threshold_comparison": PLATEAU_COMPARISON,
            "plateau_trigger_source": PLATEAU_TRIGGER,
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
            "non_swept_settings_diff": [],
            "resources": plan["resources"],
            "expected_output_root": f"runs/{EXECUTION_ID}",
            "authorization_schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "execution_authorization_v1"
            ),
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    job_path = PACKAGE_DIR / "job.json"
    _write_json(job_path, job)

    manifest = digested(
        {
            "schema": (
                "paper_i_ra_adapt_cumulative_relative_r70_"
                "package_manifest_v1"
            ),
            "status": "passed_inert_one_row",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "row_count": 1,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "control_files": [
                _package_binding(PACKAGE_DIR / name)
                for name in CONTROL_FILES
            ],
            "derived_protocol": _package_binding(
                derived_path, canonical=True
            ),
            "source_archive": _package_binding(source_archive),
            "source_archive_manifest": _package_binding(
                source_manifest_path, canonical=True
            ),
            "source_delta_receipt": _package_binding(
                source_delta_path, canonical=True
            ),
            "resume_input": _package_binding(
                resume_input_path, canonical=True
            ),
            "source_lock_audit": _package_binding(
                audit_path, canonical=True
            ),
            "execution_plan": _package_binding(plan_path, canonical=True),
            "job": _package_binding(job_path, canonical=True),
            "lineage": lineage,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submit_descriptor_present": False,
            "submitted": False,
        }
    )
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "package_id": PACKAGE_ID,
                "execution_id": EXECUTION_ID,
                "package_manifest_sha256": manifest["sha256"],
                "derived_protocol_sha256": derived["sha256"],
                "source_archive_sha256": sha256_file(source_archive),
                "resume_archive_sha256": sha256_file(resume_archive),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
