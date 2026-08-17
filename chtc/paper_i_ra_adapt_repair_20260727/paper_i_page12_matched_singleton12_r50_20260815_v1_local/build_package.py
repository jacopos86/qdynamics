#!/usr/bin/env python3
"""Materialize the inert local Page-12 matched singleton-12 package.

The six RA protocols are copied byte-for-byte from the sealed 2026-08-07
Page-12 package.  Six conventional, unwhitened Append protocols are then
materialized from those same resolved problems and guarded singleton pools.
The source archive is the sealed Page-12 archive plus one hash-pinned module
which the sealed Append implementation imports at runtime.
"""

from __future__ import annotations

import gzip
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import *  # noqa: E402,F403


BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID
GENERATED_PATHS = (
    "bundle_materialization",
    "jobs",
    "source",
    "execution_plan.json",
    "package_manifest.json",
    "queue.tsv",
)
CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _verify_file(
    path: Path,
    *,
    label: str,
    sha256: str,
    size_bytes: int | None = None,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise PackageContractError(f"{label} is absent or not a regular file.")
    if size_bytes is not None and path.stat().st_size != size_bytes:
        raise PackageContractError(f"{label} size drifted.")
    if sha256_file(path) != sha256:
        raise PackageContractError(f"{label} bytes drifted.")


def _validate_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = PARENT_PACKAGE / "package_manifest.json"
    _verify_file(
        manifest_path,
        label="sealed Page-12 package manifest",
        sha256=PARENT_PACKAGE_MANIFEST_FILE_SHA256,
    )
    manifest = load_json(manifest_path, label="sealed Page-12 package manifest")
    if (
        verify_self_digest(manifest, label="sealed Page-12 package manifest")
        != PARENT_PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids")
        != [ra_execution_id(regime, nph) for regime, nph in REGIME_ROWS]
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Sealed Page-12 package identity drifted.")

    bundle_path = PARENT_BUNDLE / "bundle_manifest.json"
    locks_path = PARENT_BUNDLE / "source_locks.json"
    source_manifest_path = PARENT_PACKAGE / "source/source_archive_manifest.json"
    archive_path = PARENT_PACKAGE / "source/source_locked.tar.gz"
    _verify_file(
        bundle_path,
        label="sealed Page-12 bundle manifest",
        sha256=PARENT_BUNDLE_MANIFEST_FILE_SHA256,
    )
    _verify_file(
        locks_path,
        label="sealed Page-12 source locks",
        sha256=PARENT_SOURCE_LOCKS_FILE_SHA256,
    )
    _verify_file(
        source_manifest_path,
        label="sealed Page-12 source manifest",
        sha256=PARENT_SOURCE_MANIFEST_FILE_SHA256,
    )
    _verify_file(
        archive_path,
        label="sealed Page-12 source archive",
        sha256=PARENT_SOURCE_ARCHIVE_SHA256,
        size_bytes=PARENT_SOURCE_ARCHIVE_SIZE_BYTES,
    )
    bundle = load_json(bundle_path, label="sealed Page-12 bundle manifest")
    locks = load_json(locks_path, label="sealed Page-12 source locks")
    source_manifest = load_json(
        source_manifest_path, label="sealed Page-12 source manifest"
    )
    if (
        verify_self_digest(bundle, label="sealed Page-12 bundle manifest")
        != PARENT_BUNDLE_MANIFEST_CANONICAL_SHA256
        or verify_self_digest(locks, label="sealed Page-12 source locks")
        != PARENT_SOURCE_LOCKS_CANONICAL_SHA256
        or verify_self_digest(source_manifest, label="sealed Page-12 source manifest")
        != PARENT_SOURCE_MANIFEST_CANONICAL_SHA256
        or locks.get("implementation_sources", {}).get("sha256")
        != PARENT_IMPLEMENTATION_SOURCE_INVENTORY_SHA256
        or source_manifest.get("archive") != manifest.get("source_archive")
        or source_manifest.get("member_count") != len(source_manifest.get("members", []))
    ):
        raise PackageContractError("Sealed Page-12 source authority drifted.")
    return manifest, locks, source_manifest


def _safe_extract_parent(
    *, source_manifest: Mapping[str, Any], destination: Path
) -> list[dict[str, Any]]:
    rows = source_manifest.get("members")
    if not isinstance(rows, list):
        raise PackageContractError("Sealed source member closure is absent.")
    expected = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(expected) != len(rows):
        raise PackageContractError("Sealed source member closure is ambiguous.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    archive_path = PARENT_PACKAGE / "source/source_locked.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(member.name, label="tar member").as_posix()
            row = expected.get(relative)
            if (
                row is None
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(f"Unsafe sealed source member: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(f"Unreadable sealed source member: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            _verify_file(
                target,
                label=f"extracted source member {relative}",
                sha256=str(row.get("sha256")),
                size_bytes=int(row.get("size_bytes", -1)),
            )
            observed.add(relative)
    if observed != set(expected):
        raise PackageContractError("Sealed source extraction is incomplete.")
    return [dict(expected[path]) for path in sorted(expected)]


def _derived_source(
    *, source_manifest: Mapping[str, Any], temporary_root: Path
) -> tuple[Path, dict[str, Any]]:
    source_root = temporary_root / "source_root"
    rows = _safe_extract_parent(
        source_manifest=source_manifest, destination=source_root
    )
    dependency = REPO_ROOT / APPEND_RUNTIME_DEPENDENCY
    _verify_file(
        dependency,
        label="historical Append runtime dependency",
        sha256=APPEND_RUNTIME_DEPENDENCY_SHA256,
        size_bytes=APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES,
    )
    target = source_root / APPEND_RUNTIME_DEPENDENCY
    if target.exists():
        _verify_file(
            target,
            label="sealed Append runtime dependency",
            sha256=APPEND_RUNTIME_DEPENDENCY_SHA256,
            size_bytes=APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES,
        )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dependency, target)
        rows.append(
            {
                "path": APPEND_RUNTIME_DEPENDENCY.as_posix(),
                "sha256": APPEND_RUNTIME_DEPENDENCY_SHA256,
                "size_bytes": APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES,
                "source_kind": "hash_pinned_historical_append_runtime_dependency",
            }
        )
    resume_rows = [
        row for row in rows if row.get("path") == SEALED_RESUME_READER.as_posix()
    ]
    if (
        len(resume_rows) != 1
        or resume_rows[0].get("sha256") != SEALED_RESUME_READER_SHA256
        or int(resume_rows[0].get("size_bytes", -1))
        != SEALED_RESUME_READER_SIZE_BYTES
    ):
        raise PackageContractError("Sealed checkpoint resume reader drifted.")
    _verify_file(
        source_root / SEALED_RESUME_READER,
        label="sealed checkpoint resume reader",
        sha256=SEALED_RESUME_READER_SHA256,
        size_bytes=SEALED_RESUME_READER_SIZE_BYTES,
    )
    rows = sorted(rows, key=lambda row: str(row["path"]))
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT
            ) as archive:
                for row in rows:
                    path = source_root / str(row["path"])
                    _verify_file(
                        path,
                        label=f"derived source member {row['path']}",
                        sha256=str(row["sha256"]),
                        size_bytes=int(row["size_bytes"]),
                    )
                    info = tarfile.TarInfo(str(row["path"]))
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    manifest = digested(
        {
            "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA,
            "status": "passed",
            "package_id": PACKAGE_ID,
            "parent_source_archive_sha256": PARENT_SOURCE_ARCHIVE_SHA256,
            "parent_implementation_source_inventory_sha256": (
                PARENT_IMPLEMENTATION_SOURCE_INVENTORY_SHA256
            ),
            "additive_dependency": {
                "path": APPEND_RUNTIME_DEPENDENCY.as_posix(),
                "sha256": APPEND_RUNTIME_DEPENDENCY_SHA256,
                "size_bytes": APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES,
            },
            "member_count": len(rows),
            "members": rows,
            "archive": binding(archive_path, root=PACKAGE_DIR),
            "archive_construction_no_ambient_repo_imports": True,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "post_extraction_overlay_count": 1,
            "sealed_resume_reader": {
                "path": SEALED_RESUME_READER.as_posix(),
                "sha256": SEALED_RESUME_READER_SHA256,
                "size_bytes": SEALED_RESUME_READER_SIZE_BYTES,
                "ambient_resume_overlay": False,
            },
        }
    )
    _write_json(PACKAGE_DIR / "source/source_archive_manifest.json", manifest)
    return source_root, manifest


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if name == "pipelines" or name.startswith("pipelines.") or name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if not (
            (Path(item or ".").resolve() / "pipelines").exists()
            or (Path(item or ".").resolve() / "src").exists()
        )
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    module = importlib.import_module("pipelines.static_adapt.ra_adapt")
    try:
        Path(str(module.__file__)).resolve().relative_to(root)
    except ValueError as exc:
        raise PackageContractError("Builder import escaped the derived source.") from exc


def _append_route_contract() -> dict[str, Any]:
    return digested(
        {
            "schema": "paper_i_page12_matched_conventional_append_route_v1",
            "route_id": APPEND_ROUTE_ID,
            "selector_identity": APPEND_SELECTOR_ID,
            "selector_scope": APPEND_SELECTOR_SCOPE,
            "candidate_representation": CANDIDATE_REPRESENTATION,
            "adapter_id": APPEND_ADAPTER_ID,
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "ra_staged_funnel_invoked": False,
            "whitening_active": False,
            "append_position_only": True,
        }
    )


def _bundle_cell(execution: str, regime: str, nph: int, method: str) -> Any:
    from pipelines.static_adapt.ra_adapt.bundles import BundleCellSpec

    return BundleCellSpec(
        cell_id=execution,
        stage=f"page12_matched_{regime}_{method}_candidate",
        regime_id=regime,
        nph=nph,
        route_id=RA_ROUTE_ID if method == "ra_singleton_plateau" else APPEND_ROUTE_ID,
        algorithm_id=RA_ALGORITHM_ID if method == "ra_singleton_plateau" else APPEND_ALGORITHM_ID,
        selector_family="ra_adapt" if method == "ra_singleton_plateau" else "append_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION,
        horizon=TARGET_HORIZON,
        source_lock_id=source_lock_id(regime, nph),
    )


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import resolve_problem_context

    receipt = protocol.problem
    return resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=None if receipt.n_fermions is None else int(receipt.n_fermions),
        )
    )


def _append_request() -> Any:
    from pipelines.static_adapt.ra_adapt.adapters import SinglePauliWordCandidateAdapter
    from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
    from pipelines.static_adapt.sr_snake.contracts import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRExecutionPolicy,
        SRObservationPolicy,
        SRStopPolicy,
    )

    return AppendAdaptRequest(
        adapter=SinglePauliWordCandidateAdapter(),
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=TARGET_HORIZON),
            resume=FreshStart(),
        ),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=Path("checkpoints/current.json"),
                every_controller_rounds=1,
                keep_history_tail=1,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=Path("result/estimator_ledger.json")
            ),
            resource_rounds=(TARGET_HORIZON,),
        ),
    )


def _source_refs(locks: Mapping[str, Any], cell: Any) -> dict[str, str]:
    from pipelines.static_adapt.ra_adapt.bundles import _source_lock_refs

    refs = _source_lock_refs(locks, cell=cell)
    if refs.get("cell_source_lock_id") != cell.source_lock_id:
        raise PackageContractError("Cell source-lock authority drifted.")
    return refs


def _resources(nph: int) -> dict[str, Any]:
    return {
        "basis": "native_local_matched_pair_same_runtime_v1",
        "request_cpus": 4,
        "request_memory_mb": 49_152 if nph == 3 else 98_304,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
    }


def build() -> dict[str, Any]:
    if any((PACKAGE_DIR / relative).exists() for relative in GENERATED_PATHS):
        raise FileExistsError("Refusing to overwrite an existing package seal.")
    for relative in CONTROL_FILES:
        if not (PACKAGE_DIR / relative).is_file():
            raise PackageContractError(f"Missing control file: {relative}")

    parent_manifest, source_locks, parent_source_manifest = _validate_parent()
    dependency = REPO_ROOT / APPEND_RUNTIME_DEPENDENCY
    _verify_file(
        dependency,
        label="historical Append runtime dependency",
        sha256=APPEND_RUNTIME_DEPENDENCY_SHA256,
        size_bytes=APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES,
    )

    with tempfile.TemporaryDirectory(prefix="paper-i-page12-matched12-build.") as temp:
        source_root, source_manifest = _derived_source(
            source_manifest=parent_source_manifest,
            temporary_root=Path(temp),
        )
        _activate_source_root(source_root)

        from pipelines.static_adapt.ra_adapt.append import (
            APPEND_ADAPT_ALGORITHM_ID,
            build_resolved_append_protocol,
        )
        from pipelines.static_adapt.ra_adapt.bundles import (
            _bundle_protocol_materialization_authority,
        )
        from pipelines.static_adapt.ra_adapt.contracts import (
            resolved_ra_adapt_protocol_from_mapping,
        )

        if APPEND_ADAPT_ALGORITHM_ID != APPEND_ALGORITHM_ID:
            raise PackageContractError("Sealed Append algorithm identity drifted.")

        BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(PARENT_BUNDLE / "source_locks.json", BUNDLE_ROOT / "source_locks.json")

        cells = [
            _bundle_cell(execution_id(regime, nph, method), regime, nph, method)
            for regime, nph in REGIME_ROWS
            for method in METHODS
        ]
        expected = digested(
            {
                "schema": EXPECTED_ARTIFACTS_SCHEMA,
                "bundle_id": BUNDLE_ID,
                "cells": {
                    cell.cell_id: {
                        "expected_run_artifacts": expected_run_artifacts(cell.cell_id)
                    }
                    for cell in cells
                },
            }
        )
        _write_json(BUNDLE_ROOT / "expected_artifacts.json", expected)
        bundle_manifest = digested(
            {
                "schema": BUNDLE_MANIFEST_SCHEMA,
                "bundle_id": BUNDLE_ID,
                "campaign_id": CAMPAIGN_ID,
                "run_class": RUN_CLASS,
                "cell_count": 12,
                "cells": [cell.to_dict() for cell in cells],
                "parent_ra_bundle_id": PARENT_BUNDLE_ID,
                "parent_ra_bundle_manifest_sha256": PARENT_BUNDLE_MANIFEST_CANONICAL_SHA256,
                "source_locks_sha256": PARENT_SOURCE_LOCKS_CANONICAL_SHA256,
                "expected_artifacts_sha256": expected["sha256"],
                "target_horizon": TARGET_HORIZON,
                "execution_authorized": False,
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        _write_json(BUNDLE_ROOT / "bundle_manifest.json", bundle_manifest)

        append_route = _append_route_contract()
        protocols: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        queue_rows: list[str] = []
        matched_pool_rows: list[dict[str, Any]] = []
        for regime, nph in REGIME_ROWS:
            ra_execution = ra_execution_id(regime, nph)
            parent_protocol_path = PARENT_BUNDLE / "protocols" / f"{ra_execution}.json"
            ra_payload = load_json(parent_protocol_path, label=f"sealed RA protocol {regime}")
            if (
                verify_self_digest(ra_payload, label=f"sealed RA protocol {regime}")
                != RA_PROTOCOL_SHA256_BY_REGIME[regime]
                or ra_payload.get("algorithm_id") != RA_ALGORITHM_ID
                or ra_payload.get("route_contract", {}).get("sha256") != RA_ROUTE_CONTRACT_SHA256
            ):
                raise PackageContractError(f"Sealed RA protocol drifted for {regime}.")
            typed_ra = resolved_ra_adapt_protocol_from_mapping(ra_payload)
            problem = _problem_from_protocol(typed_ra)

            append_execution = append_execution_id(regime, nph)
            append_cell = _bundle_cell(
                append_execution, regime, nph, "append_singleton"
            )
            refs = _source_refs(source_locks, append_cell)
            authority = _bundle_protocol_materialization_authority(
                cell=append_cell,
                bundle_id=BUNDLE_ID,
                bundle_manifest_sha256=bundle_manifest["sha256"],
                source_locks_sha256=PARENT_SOURCE_LOCKS_CANONICAL_SHA256,
                source_lock_refs=refs,
                active_gradient_policy=ACTIVE_GRADIENT_POLICY,
                resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
            )
            append_protocol = build_resolved_append_protocol(
                problem, _append_request(), materialization_authority=authority
            )
            if (
                append_protocol.problem != typed_ra.problem
                or append_protocol.parent_inventory != typed_ra.parent_inventory
                or append_protocol.executable_pool != typed_ra.executable_pool
                or append_protocol.horizon != TARGET_HORIZON
                or append_protocol.optimizer != "powell"
                or append_protocol.optimizer_maxiter != 200
                or append_protocol.seeds != {"adapt": 7, "transpiler": 7}
                or append_protocol.adapter_id != APPEND_ADAPTER_ID
                or append_protocol.selector_identity != APPEND_SELECTOR_ID
                or append_protocol.selector_scope != APPEND_SELECTOR_SCOPE
                or append_protocol.active_gradient_policy != ACTIVE_GRADIENT_POLICY
                or append_protocol.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
                or append_protocol.lineage_authority.get("ra_staged_funnel_invoked") is not False
            ):
                raise PackageContractError(
                    f"Matched conventional Append protocol drifted for {regime}."
                )
            matched_pool_rows.append(
                {
                    "regime_id": regime,
                    "nph": nph,
                    "problem_request_sha256": typed_ra.problem.problem_request_sha256,
                    "parent_pool_sha256": typed_ra.parent_inventory.sha256,
                    "parent_pool_count": typed_ra.parent_inventory.count,
                    "executable_pool_sha256": typed_ra.executable_pool.sha256,
                    "executable_pool_count": typed_ra.executable_pool.count,
                    "ra_protocol_sha256": typed_ra.sha256,
                    "append_protocol_sha256": append_protocol.sha256,
                }
            )

            protocol_payloads = {
                "ra_singleton_plateau": ra_payload,
                "append_singleton": append_protocol.to_dict(),
            }
            for method in METHODS:
                execution = execution_id(regime, nph, method)
                protocol_path = BUNDLE_ROOT / "protocols" / f"{execution}.json"
                if method == "ra_singleton_plateau":
                    protocol_path.parent.mkdir(parents=True, exist_ok=True)
                    with protocol_path.open("xb") as stream:
                        stream.write(parent_protocol_path.read_bytes())
                    route_id = RA_ROUTE_ID
                    route_sha = RA_ROUTE_CONTRACT_SHA256
                    algorithm = RA_ALGORITHM_ID
                    adapter_id = RA_ADAPTER_ID
                    bundle_binding_sha = PARENT_BUNDLE_MANIFEST_CANONICAL_SHA256
                    entrypoint = "run_ra_adapt"
                else:
                    _write_json(protocol_path, protocol_payloads[method])
                    route_id = APPEND_ROUTE_ID
                    route_sha = append_route["sha256"]
                    algorithm = APPEND_ALGORITHM_ID
                    adapter_id = APPEND_ADAPTER_ID
                    bundle_binding_sha = bundle_manifest["sha256"]
                    entrypoint = "run_append_adapt"
                protocol_binding = {
                    "execution_id": execution,
                    "method": method,
                    **binding(protocol_path, root=PACKAGE_DIR, canonical=True),
                }
                protocols.append(protocol_binding)
                job = digested(
                    {
                        "schema": JOB_SCHEMA,
                        "package_id": PACKAGE_ID,
                        "campaign_id": CAMPAIGN_ID,
                        "bundle_id": BUNDLE_ID,
                        "execution_id": execution,
                        "cell_id": execution,
                        "regime_id": regime,
                        "method": method,
                        "num_sites": 2,
                        "nph": nph,
                        "target_horizon": TARGET_HORIZON,
                        "algorithm_id": algorithm,
                        "route_id": route_id,
                        "route_contract_sha256": route_sha,
                        "candidate_representation": CANDIDATE_REPRESENTATION,
                        "candidate_adapter_id": adapter_id,
                        "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                        "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                        "execution_entrypoint": entrypoint,
                        "protocol_path": protocol_binding["path"],
                        "protocol_file_sha256": protocol_binding["sha256"],
                        "protocol_sha256": protocol_binding["canonical_sha256"],
                        "protocol_bundle_manifest_sha256": bundle_binding_sha,
                        "source_locks_sha256": PARENT_SOURCE_LOCKS_CANONICAL_SHA256,
                        "implementation_source_inventory_sha256": PARENT_IMPLEMENTATION_SOURCE_INVENTORY_SHA256,
                        "expected_artifacts_manifest_sha256": expected["sha256"],
                        "expected_run_artifacts": expected_run_artifacts(execution),
                        "resources": _resources(nph),
                        "fresh_start_contract": {
                            "kind": "fresh_start",
                            "source_checkpoint": None,
                            "resume_archive": None,
                            "fresh_start_only": True,
                            "checkpoint_resume_authorized": False,
                        },
                        "checkpoint_observation": {
                            "every_controller_rounds": 1,
                            "keep_history_tail": 1,
                            "compact_only": True,
                            "usage": CHECKPOINT_USAGE,
                            "resume_consumable": False,
                        },
                        "execution_authorized": False,
                        "submission_authorized": False,
                        "paper_adoption_authorized": False,
                        "paper_evidence_adoption_authorized": False,
                        "submitted": False,
                    }
                )
                job_path = PACKAGE_DIR / "jobs" / f"{execution}.json"
                _write_json(job_path, job)
                jobs.append(
                    {
                        "execution_id": execution,
                        "method": method,
                        **binding(job_path, root=PACKAGE_DIR, canonical=True),
                    }
                )

        job_by_execution = {row["execution_id"]: row for row in jobs}
        for regime, nph in PAIR_EXECUTION_ORDER:
            for method in METHODS:
                execution = execution_id(regime, nph, method)
                row = job_by_execution[execution]
                queue_rows.append(
                    "\t".join(
                        (
                            execution,
                            str(row["path"]),
                            method,
                            str(nph),
                            str(_resources(nph)["max_runtime_seconds"]),
                        )
                    )
                )

        validation = digested(
            {
                "schema": VALIDATION_REPORT_SCHEMA,
                "status": "passed_inert_matched_singleton12",
                "bundle_id": BUNDLE_ID,
                "protocol_count": 12,
                "method_counts": {"ra_singleton_plateau": 6, "append_singleton": 6},
                "target_horizon": TARGET_HORIZON,
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "seeds": {"adapt": 7, "transpiler": 7},
                "same_resolved_problem_and_pool_per_pair": True,
                "append_whitening_active": False,
                "append_ra_staged_funnel_invoked": False,
                "compact_checkpoint_keep_history_tail": 1,
                "checkpoint_usage": CHECKPOINT_USAGE,
                "fresh_start_only": True,
                "checkpoint_resume_authorized": False,
                "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
                "matched_pool_receipts": matched_pool_rows,
                "ra_protocol_sha256_by_regime": dict(RA_PROTOCOL_SHA256_BY_REGIME),
                "scientific_execution_performed": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json(BUNDLE_ROOT / "validation_report.json", validation)

        queue_path = PACKAGE_DIR / "queue.tsv"
        with queue_path.open("xb") as stream:
            stream.write(("\n".join(queue_rows) + "\n").encode("utf-8"))
        plan = digested(
            {
                "schema": EXECUTION_PLAN_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "row_count": 12,
                "execution_order": [
                    execution_id(regime, nph, method)
                    for regime, nph in PAIR_EXECUTION_ORDER
                    for method in METHODS
                ],
                "pair_order": [
                    {"regime_id": regime, "nph": nph, "methods": list(METHODS)}
                    for regime, nph in PAIR_EXECUTION_ORDER
                ],
                "maximum_concurrency": 1,
                "strongest_matched_pairs_first": True,
                "same_native_runtime_per_pair": True,
                "source_archive_sha256": source_manifest["archive"]["sha256"],
                "execution_source_policy": EXECUTION_SOURCE_POLICY,
                "post_extraction_overlay_count": 1,
                "fresh_start_only": True,
                "checkpoint_usage": CHECKPOINT_USAGE,
                "checkpoint_resume_authorized": False,
                "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
                "execution_authorized": False,
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        _write_json(PACKAGE_DIR / "execution_plan.json", plan)

        manifest = digested(
            {
                "schema": PACKAGE_MANIFEST_SCHEMA,
                "status": "passed_inert_matched_singleton12",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "run_class": RUN_CLASS,
                "row_count": 12,
                "execution_ids": list(expected_execution_ids()),
                "execution_order": plan["execution_order"],
                "methods": list(METHODS),
                "regimes": [
                    {"regime_id": regime, "nph": nph} for regime, nph in REGIME_ROWS
                ],
                "parent_page12_package": {
                    "path": PARENT_PACKAGE_RELATIVE.as_posix(),
                    "package_manifest_sha256": PARENT_PACKAGE_MANIFEST_CANONICAL_SHA256,
                    "bundle_manifest_sha256": PARENT_BUNDLE_MANIFEST_CANONICAL_SHA256,
                    "source_archive_sha256": PARENT_SOURCE_ARCHIVE_SHA256,
                },
                "bundle_manifest": binding(
                    BUNDLE_ROOT / "bundle_manifest.json", root=PACKAGE_DIR, canonical=True
                ),
                "bundle_expected_artifacts": binding(
                    BUNDLE_ROOT / "expected_artifacts.json", root=PACKAGE_DIR, canonical=True
                ),
                "bundle_source_locks": binding(
                    BUNDLE_ROOT / "source_locks.json", root=PACKAGE_DIR, canonical=True
                ),
                "bundle_validation_report": binding(
                    BUNDLE_ROOT / "validation_report.json", root=PACKAGE_DIR, canonical=True
                ),
                "protocols": protocols,
                "jobs": jobs,
                "source_archive": source_manifest["archive"],
                "source_archive_manifest": binding(
                    PACKAGE_DIR / "source/source_archive_manifest.json",
                    root=PACKAGE_DIR,
                    canonical=True,
                ),
                "queue": binding(queue_path, root=PACKAGE_DIR),
                "execution_plan": binding(
                    PACKAGE_DIR / "execution_plan.json", root=PACKAGE_DIR, canonical=True
                ),
                "control_files": [
                    binding(PACKAGE_DIR / relative, root=PACKAGE_DIR)
                    for relative in CONTROL_FILES
                ],
                "operational_checkpoint_overlay": {
                    "relative_path": OPERATIONAL_CHECKPOINT_OVERLAY.as_posix(),
                    "sealed_sha256": SEALED_CHECKPOINT_SHA256,
                    "candidate_sha256": OPERATIONAL_CHECKPOINT_OVERLAY_SHA256,
                    "execution_source_policy": EXECUTION_SOURCE_POLICY,
                    "post_extraction_overlay_count": 1,
                    "ambient_resume_overlay": False,
                    "sealed_resume_reader_path": SEALED_RESUME_READER.as_posix(),
                    "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
                    "fresh_start_only": True,
                    "checkpoint_usage": CHECKPOINT_USAGE,
                    "checkpoint_resume_authorized": False,
                    "parity_canary_scope": PARITY_CANARY_SCOPE,
                    "multi_round_compact_tail_resume_validated": False,
                    "scientific_parity_receipt_file_sha256": STRONG5_PARITY_RECEIPT_FILE_SHA256,
                    "scientific_parity_receipt_canonical_sha256": STRONG5_PARITY_RECEIPT_CANONICAL_SHA256,
                },
                "target_horizon": TARGET_HORIZON,
                "activation_artifacts_present": False,
                "authorizations_present": False,
                "execution_authorized": False,
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submission_ready": False,
                "submitted": False,
                "scientific_execution_performed": False,
            }
        )
        _write_json(PACKAGE_DIR / "package_manifest.json", manifest)
        return {
            "status": manifest["status"],
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "source_archive_sha256": source_manifest["archive"]["sha256"],
            "row_count": 12,
        }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
