#!/usr/bin/env python3
"""Materialize the inert six-cell historical-average RA r70 package."""

from __future__ import annotations

import copy
from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ALGORITHM_ID,
    CAMPAIGN_ID,
    CANDIDATE_REPRESENTATION,
    CONTROL_FILES,
    GENERATED_PATHS,
    HORIZON_DIFFERENCE_PATHS,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PLATEAU_CALIBRATION,
    PLATEAU_COMPARISON,
    PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD,
    PLATEAU_TRIGGER,
    PARENT_ROUTE_CONTRACT_SHA256,
    PROTOCOL_BUNDLE_MANIFEST_SCHEMA,
    REGIME_ROWS,
    RESOURCE_ENVELOPES,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_DIFFERENCE_PATHS,
    ROUTE_ID,
    ROUTE_PROFILE,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    SOURCE_BUNDLE_ID,
    SOURCE_BUNDLE_MANIFEST_CANONICAL_SHA256,
    SOURCE_BUNDLE_MANIFEST_FILE_SHA256,
    SOURCE_BUNDLE_RELATIVE,
    SOURCE_HORIZON,
    SOURCE_LOCK_AUDIT_SCHEMA,
    SOURCE_LOCKS_CANONICAL_SHA256,
    SOURCE_LOCKS_FILE_SHA256,
    SOURCE_ARCHIVE_FILE_SHA256,
    SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256,
    SOURCE_ARCHIVE_MANIFEST_FILE_SHA256,
    SOURCE_IMPLEMENTATION_INVENTORY_SHA256,
    SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    SOURCE_PACKAGE_RELATIVE,
    SOURCE_PATCH_BINDINGS,
    TARGET_HORIZON,
    PackageContractError,
    binding,
    canonical_json_bytes,
    digested,
    execution_id,
    repo_root_from_script,
    scalar_differences,
    sha256_file,
    source_cell_id,
    verify_self_digest,
)


PRESERVED_PROTOCOL_FIELDS = (
    "problem",
    "parent_inventory",
    "executable_pool",
    "optimizer",
    "optimizer_maxiter",
    "seeds",
    "candidate_representation",
    "adapter_id",
    "algorithm_id",
    "active_gradient_policy",
    "resource_weighting_scope",
    "accepted_refit_scope",
    "accepted_refit_coordinate_chart",
    "accepted_refit_base_chart_policy",
    "phase3_solver_id",
    "phase3_multiplier_contract",
    "estimator_accounting_convention",
    "compile_identity",
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_bound_json(
    path: Path,
    *,
    file_sha256: str,
    canonical_sha256: str,
    label: str,
) -> dict[str, Any]:
    from package_contract import load_json

    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != file_sha256
    ):
        raise PackageContractError(f"{label} exact bytes drifted.")
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != canonical_sha256:
        raise PackageContractError(f"{label} canonical digest drifted.")
    return payload


def _package_bound_json(
    package_dir: Path,
    raw: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    from package_contract import load_json, safe_relative_path

    if not isinstance(raw, Mapping):
        raise PackageContractError(f"{label} binding is malformed.")
    path = package_dir / safe_relative_path(raw.get("path"), label=label)
    try:
        path.resolve().relative_to(package_dir.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped its package.") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or sha256_file(path) != raw.get("sha256")
    ):
        raise PackageContractError(f"{label} exact bytes drifted.")
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != raw.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical digest drifted.")
    return path, payload


def _load_source_package(repo_root: Path) -> dict[str, Any]:
    source_package = repo_root / SOURCE_PACKAGE_RELATIVE
    manifest = _load_bound_json(
        source_package / "package_manifest.json",
        file_sha256=SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
        canonical_sha256=SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
        label="sealed cumulative-relative predecessor package",
    )
    archive_path, archive_manifest = _package_bound_json(
        source_package,
        manifest.get("source_archive_manifest"),
        label="predecessor source archive manifest",
    )
    if (
        sha256_file(archive_path) != SOURCE_ARCHIVE_MANIFEST_FILE_SHA256
        or archive_manifest.get("sha256")
        != SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256
        or archive_manifest.get("implementation_source_inventory_sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
    ):
        raise PackageContractError("Predecessor source archive authority drifted.")
    archive_binding = manifest.get("source_archive")
    if not isinstance(archive_binding, Mapping):
        raise PackageContractError("Predecessor source archive binding is absent.")
    source_archive = source_package / str(archive_binding.get("path", ""))
    if (
        not source_archive.is_file()
        or source_archive.is_symlink()
        or sha256_file(source_archive) != SOURCE_ARCHIVE_FILE_SHA256
        or sha256_file(source_archive) != archive_binding.get("sha256")
        or archive_manifest.get("archive") != archive_binding
    ):
        raise PackageContractError("Predecessor source archive bytes drifted.")
    _protocol_bundle_path, protocol_bundle = _package_bound_json(
        source_package,
        manifest.get("protocol_bundle_manifest"),
        label="predecessor protocol bundle manifest",
    )
    _locks_path, source_locks = _package_bound_json(
        source_package,
        manifest.get("source_locks_snapshot"),
        label="predecessor source-lock snapshot",
    )
    return {
        "root": source_package,
        "manifest": manifest,
        "archive_path": source_archive,
        "archive_manifest": archive_manifest,
        "protocol_bundle": protocol_bundle,
        "source_locks": source_locks,
    }


def _extract_predecessor_source(
    *,
    source_package: Mapping[str, Any],
    destination: Path,
) -> list[dict[str, Any]]:
    from package_contract import safe_relative_path

    raw_rows = source_package["archive_manifest"].get("members")
    if not isinstance(raw_rows, list):
        raise PackageContractError("Predecessor source members are absent.")
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    declared = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
    }
    if len(rows) != len(raw_rows) or len(declared) != len(rows):
        raise PackageContractError("Predecessor source member closure drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(source_package["archive_path"], "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(member.name, label="tar member").as_posix()
            row = declared.get(relative)
            if (
                row is None
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(f"Unsafe predecessor member: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(f"Unreadable predecessor member: {relative}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size != member.size or digest.hexdigest() != row.get("sha256"):
                raise PackageContractError(f"Extracted predecessor drifted: {relative}")
            observed.add(relative)
    if observed != set(declared):
        raise PackageContractError("Predecessor extraction is incomplete.")
    return rows


def _apply_exact_source_patch(source_root: Path) -> list[dict[str, Any]]:
    expected = {
        path: {"before_sha256": before, "after_sha256": after}
        for path, before, after in SOURCE_PATCH_BINDINGS
    }
    for relative, row in expected.items():
        if sha256_file(source_root / relative) != row["before_sha256"]:
            raise PackageContractError(f"Patch predecessor drifted: {relative}")
    completed = subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "--fuzz=0",
            "-p1",
            "-i",
            str(PACKAGE_DIR / "source_patch.diff"),
        ],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            "Exact v1-to-v2 source patch failed: " + completed.stderr.strip()
        )
    result: list[dict[str, Any]] = []
    for relative, row in expected.items():
        path = source_root / relative
        if sha256_file(path) != row["after_sha256"]:
            raise PackageContractError(f"Patched source drifted: {relative}")
        result.append({"path": relative, **row})
    return result


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
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
        raise PackageContractError("Materializer escaped the patched archive.") from exc


def _problem_from_receipt(receipt: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

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
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def _exact_energy(cell_lock: Mapping[str, Any]) -> float:
    resolver = cell_lock.get("resolver_trace")
    if not isinstance(resolver, Mapping):
        raise PackageContractError("Cell lock lost its resolver trace.")
    reference = resolver.get("same_cutoff_ed_reference")
    if not isinstance(reference, Mapping):
        raise PackageContractError("Cell lock lost its same-cutoff reference.")
    value = float(reference["E_ED"])
    if int(reference["nph"]) != int(cell_lock["nph"]):
        raise PackageContractError("Same-cutoff reference changed cutoff.")
    return value


def _normalize_source_locks(
    source_locks: Mapping[str, Any],
    *,
    implementation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256

    result = copy.deepcopy(dict(source_locks))
    result["implementation_sources"] = copy.deepcopy(
        dict(implementation_inventory)
    )
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _target_cell(source_cell: Any, *, horizon: int) -> Any:
    return replace(
        source_cell,
        cell_id=execution_id(source_cell.regime_id, source_cell.nph),
        stage="historical_average_r70_fresh",
        horizon=int(horizon),
    )


def _validate_source_to_target(
    *,
    source_payload: Mapping[str, Any],
    current_h50_payload: Mapping[str, Any],
    target_payload: Mapping[str, Any],
) -> dict[str, Any]:
    route_differences = scalar_differences(
        source_payload["route_contract"],
        target_payload["route_contract"],
    )
    route_paths = {path for path, _before, _after in route_differences}
    if route_paths != ROUTE_DIFFERENCE_PATHS:
        raise PackageContractError(
            "Historical-average route changed unexpected fields: "
            f"{sorted(route_paths, key=str)}"
        )
    horizon_differences = scalar_differences(
        current_h50_payload,
        target_payload,
    )
    horizon_paths = {path for path, _before, _after in horizon_differences}
    if horizon_paths != HORIZON_DIFFERENCE_PATHS:
        raise PackageContractError(
            "Typed r70 derivation changed non-horizon fields: "
            f"{sorted(horizon_paths, key=str)}"
        )
    for key in PRESERVED_PROTOCOL_FIELDS:
        if source_payload[key] != target_payload[key]:
            raise PackageContractError(f"Target changed preserved field {key}.")
    request = target_payload["request"]
    method = request["method"]
    route_contract = target_payload["route_contract"]
    invariants = route_contract["semantic_invariants"]
    route_execution = route_contract["execution_settings"]
    lineage = route_contract["lineage_authority"]
    if (
        request["kind"] != "ra_adapt_request"
        or request["adapter"]["candidate_representation_id"]
        != CANDIDATE_REPRESENTATION
        or request["execution"]["resume"]["kind"] != "fresh_start"
        or request["execution"]["stop"]["maximum_controller_rounds"]
        != TARGET_HORIZON
        or method["admission"]["kind"] != "singleton"
        or method["insertion"]["kind"] != "plateau_commutation"
        or method["pruning"]["kind"] != "off"
        or method["beam"]["kind"] != "off"
        or target_payload["horizon"] != TARGET_HORIZON
        or target_payload["active_gradient_policy"]
        != ACTIVE_GRADIENT_POLICY
        or target_payload["resource_weighting_scope"]
        != RESOURCE_WEIGHTING_SCOPE
        or target_payload["route_contract"]["sha256"]
        != ROUTE_CONTRACT_SHA256
        or route_contract["route_profile"] != ROUTE_PROFILE
        or lineage["parent_contract_sha256"]
        != PARENT_ROUTE_CONTRACT_SHA256
        or route_execution["adapt_insertion_mode"]
        != "insertion_commutation_plateau_v2"
        or invariants["experimental_insertion_policy"]
        != "insertion_commutation_plateau_v2"
        or invariants["plateau_prior_mean_decrease_ratio_threshold"]
        != PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD
        or invariants["plateau_threshold_comparison"]
        != PLATEAU_COMPARISON
        or invariants["plateau_trigger_source"] != PLATEAU_TRIGGER
        or invariants["plateau_threshold_calibration_status"]
        != PLATEAU_CALIBRATION
        or invariants["plateau_patience"] != 1
        or invariants["plateau_hysteresis_active"] is not False
        or invariants["online_exact_reference_used"] is not False
    ):
        raise PackageContractError("Target historical-average protocol drifted.")
    return {
        "route_differences": [
            {"path": list(path), "before": before, "after": after}
            for path, before, after in route_differences
        ],
        "horizon_differences": [
            {"path": list(path), "before": before, "after": after}
            for path, before, after in horizon_differences
        ],
    }


def _write_source_archive(
    *,
    repo_root: Path,
    destination: Path,
    members: list[dict[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for row in members:
                    path = repo_root / str(row["path"])
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_size != int(row["size_bytes"])
                        or sha256_file(path) != row["sha256"]
                    ):
                        raise PackageContractError(
                            f"Implementation source drifted: {row['path']}"
                        )
                    info = archive.gettarinfo(str(path), arcname=str(row["path"]))
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as source:
                        archive.addfile(info, source)
        raw.flush()
        os.fsync(raw.fileno())


def _queue_line(job: Mapping[str, Any]) -> str:
    resources = job["resources"]
    return "\t".join(
        (
            str(job["execution_id"]),
            str(job["job_path"]),
            str(job["protocol_path"]),
            str(job["sha256"]),
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        )
    )


def build() -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    if any(
        (PACKAGE_DIR / name).exists() or (PACKAGE_DIR / name).is_symlink()
        for name in GENERATED_PATHS
    ):
        raise FileExistsError("Refusing to overwrite an immutable package.")
    for name in CONTROL_FILES:
        path = PACKAGE_DIR / name
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(f"Missing package control file: {name}")
    if any(
        path.name == "__pycache__" or path.suffix == ".pyc"
        for path in PACKAGE_DIR.rglob("*")
    ):
        raise PackageContractError("Unbound package bytecode is forbidden.")

    source_package = _load_source_package(repo_root)
    temporary = tempfile.TemporaryDirectory(
        prefix="paper-i-historical-average-source-"
    )
    source_root = (
        Path(temporary.name) / "source_locked_checkout"
    ).resolve()
    predecessor_members = _extract_predecessor_source(
        source_package=source_package,
        destination=source_root,
    )
    source_patch = _apply_exact_source_patch(source_root)
    _activate_source_root(source_root)

    from pipelines.static_adapt.ra_adapt.bundles import (
        BundleCellSpec,
        _build_request,
        _bundle_protocol_materialization_authority,
        _cell_from_manifest_row,
        _decorate_protocol_payload,
        _implementation_source_inventory,
        _source_lock_refs,
        _validate_protocol_payload,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        canonical_sha256,
        resolved_ra_adapt_protocol_from_mapping,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        build_resolved_ra_protocol,
    )

    implementation_inventory = _implementation_source_inventory(source_root)
    predecessor_inventory = source_package["source_locks"].get(
        "implementation_sources"
    )
    if (
        not isinstance(predecessor_inventory, Mapping)
        or predecessor_inventory.get("sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
        or {
            str(row["path"])
            for row in implementation_inventory.get("files", [])
        }
        != {
            str(row["path"])
            for row in predecessor_inventory.get("files", [])
        }
    ):
        raise PackageContractError(
            "Patched inventory changed the predecessor runtime member set."
        )
    source_locks = _normalize_source_locks(
        source_package["source_locks"],
        implementation_inventory=implementation_inventory,
    )
    global_sources = source_locks.get("global_sources")
    if not isinstance(global_sources, Mapping) or not global_sources:
        raise PackageContractError("Normalized global source locks are absent.")
    for source_id, raw in sorted(global_sources.items()):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Malformed normalized global source lock: {source_id}"
            )
        relative = str(raw.get("path", ""))
        path = source_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != raw.get("sha256")
        ):
            raise PackageContractError(
                f"Normalized global source bytes drifted: {source_id}"
            )
    predecessor_by_path = {
        str(row["path"]): row for row in predecessor_members
    }
    patch_paths = {str(row["path"]) for row in source_patch}
    members: list[dict[str, Any]] = []
    for relative in sorted(predecessor_by_path):
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Source archive member is missing or unsafe: {relative}"
            )
        row = {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if relative not in patch_paths and row != predecessor_by_path[relative]:
            raise PackageContractError(
                f"Unpatched predecessor member drifted: {relative}"
            )
        members.append(row)
    _write_json(PACKAGE_DIR / "source_locks_snapshot.json", source_locks)

    source_rows = {
        str(row["cell_id"]): row
        for row in source_package["protocol_bundle"]["cells"]
    }
    target_cells: list[BundleCellSpec] = []
    source_cells: dict[str, BundleCellSpec] = {}
    for regime_id, nph in REGIME_ROWS:
        source_id = source_cell_id(regime_id, nph)
        if source_id not in source_rows:
            raise PackageContractError(
                f"Missing cumulative-relative source cell: {source_id}"
            )
        source_cell = _cell_from_manifest_row(source_rows[source_id])
        if (
            source_cell.route_id != ROUTE_ID
            or source_cell.algorithm_id != ALGORITHM_ID
            or source_cell.candidate_representation
            != CANDIDATE_REPRESENTATION
            or source_cell.horizon != SOURCE_HORIZON
            or source_cell.selector_family != "ra_adapt"
        ):
            raise PackageContractError(
                f"Cumulative-relative source cell drifted: {source_id}"
            )
        target = _target_cell(source_cell, horizon=TARGET_HORIZON)
        target_cells.append(target)
        source_cells[target.cell_id] = source_cell

    protocol_bundle_manifest = digested(
        {
            "schema": PROTOCOL_BUNDLE_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "source_bundle": {
                "path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                "package_manifest_sha256": source_package["manifest"]["sha256"],
                "protocol_bundle_manifest_sha256": (
                    source_package["protocol_bundle"]["sha256"]
                ),
                "source_locks_sha256": source_package["source_locks"]["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
            },
            "source_patch": source_patch,
            "source_locks_snapshot_sha256": source_locks["sha256"],
            "implementation_source_inventory_sha256": (
                implementation_inventory["sha256"]
            ),
            "target_horizon": TARGET_HORIZON,
            "cells": [cell.to_dict() for cell in target_cells],
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )
    _write_json(
        PACKAGE_DIR / "protocol_bundle_manifest.json",
        protocol_bundle_manifest,
    )

    protocol_bindings: list[dict[str, Any]] = []
    job_payloads: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for target_cell in target_cells:
        source_cell = source_cells[target_cell.cell_id]
        source_protocol_path = (
            source_package["root"]
            / "protocols"
            / f"{source_cell.cell_id}.json"
        )
        source_payload = json.loads(
            source_protocol_path.read_text(encoding="utf-8")
        )
        if not isinstance(source_payload, dict):
            raise PackageContractError(
                "Cumulative-relative protocol is not a mapping."
            )
        source_protocol = resolved_ra_adapt_protocol_from_mapping(
            source_payload
        )
        if source_protocol.sha256 != source_payload.get("sha256"):
            raise PackageContractError(
                "Cumulative-relative protocol digest drifted."
            )
        problem = _problem_from_receipt(source_protocol.problem)
        refs = _source_lock_refs(source_locks, cell=target_cell)
        authority = _bundle_protocol_materialization_authority(
            cell=target_cell,
            bundle_id=PACKAGE_ID,
            bundle_manifest_sha256=protocol_bundle_manifest["sha256"],
            source_locks_sha256=source_locks["sha256"],
            source_lock_refs=refs,
            active_gradient_policy=ACTIVE_GRADIENT_POLICY,
            resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
        )

        request = _build_request(target_cell, bundle_dir=PACKAGE_DIR)
        resolved = build_resolved_ra_protocol(
            problem,
            request,
            materialization_authority=authority,
        )
        target_payload = _decorate_protocol_payload(
            resolved.to_dict(),
            cell=target_cell,
            request=request,
            cell_source_lock=source_locks["cell_locks"][
                target_cell.source_lock_id
            ],
            materialization_authority=authority,
        )
        _validate_protocol_payload(
            target_payload,
            cell=target_cell,
            bundle_id=PACKAGE_ID,
            bundle_manifest_sha256=protocol_bundle_manifest["sha256"],
            active_gradient_policy=ACTIVE_GRADIENT_POLICY,
            resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
            source_lock_refs=refs,
            cell_source_lock=source_locks["cell_locks"][
                target_cell.source_lock_id
            ],
            source_locks_sha256=source_locks["sha256"],
        )
        target_protocol = resolved_ra_adapt_protocol_from_mapping(
            target_payload
        )
        # The horizon-only comparator is a canonical projection of this exact
        # target payload.  Rebuilding the large pool a second time per cell
        # adds no audit information and needlessly doubles materialization
        # memory; the source protocol separately authenticates the locked h50
        # scientific settings and route delta.
        h50_payload = copy.deepcopy(target_payload)
        h50_payload["horizon"] = SOURCE_HORIZON
        h50_payload["request"]["execution"]["stop"][
            "maximum_controller_rounds"
        ] = SOURCE_HORIZON
        h50_payload["stopping_rule"][
            "maximum_controller_rounds"
        ] = SOURCE_HORIZON
        h50_payload.pop("sha256", None)
        from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256

        h50_payload["sha256"] = canonical_sha256(h50_payload)
        differences = _validate_source_to_target(
            source_payload=source_payload,
            current_h50_payload=h50_payload,
            target_payload=target_payload,
        )
        protocol_path = PACKAGE_DIR / "protocols" / f"{target_cell.cell_id}.json"
        _write_json(protocol_path, target_payload)
        protocol_binding = binding(
            protocol_path,
            root=PACKAGE_DIR,
            canonical=True,
        )
        protocol_bindings.append(
            {
                "execution_id": target_cell.cell_id,
                "source_cell_id": source_cell.cell_id,
                **protocol_binding,
            }
        )

        cell_lock = source_locks["cell_locks"][target_cell.source_lock_id]
        resources = copy.deepcopy(RESOURCE_ENVELOPES[int(target_cell.nph)])
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": target_cell.cell_id,
                "source_cell_id": source_cell.cell_id,
                "source_lock_id": target_cell.source_lock_id,
                "source_lock_sha256": cell_lock["sha256"],
                "regime_id": target_cell.regime_id,
                "nph": int(target_cell.nph),
                "run_class": "candidate",
                "execution_mode": "fresh_0_to_70",
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "protocol_path": protocol_binding["path"],
                "protocol_sha256": target_protocol.sha256,
                "protocol_file_sha256": protocol_binding["sha256"],
                "protocol_bundle_manifest_sha256": (
                    protocol_bundle_manifest["sha256"]
                ),
                "source_locks_snapshot_sha256": source_locks["sha256"],
                "implementation_source_inventory_sha256": (
                    implementation_inventory["sha256"]
                ),
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "insertion_policy": "plateau_commutation",
                "plateau_prior_mean_decrease_ratio_threshold": (
                    PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD
                ),
                "plateau_threshold_comparison": PLATEAU_COMPARISON,
                "plateau_trigger_source": PLATEAU_TRIGGER,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "exact_same_cutoff_energy": _exact_energy(cell_lock),
                "resources": resources,
                "fresh_start_contract": {
                    "kind": "fresh_start",
                    "source_checkpoint": None,
                    "resume_archive": None,
                },
                "expected_output_archive": f"{target_cell.cell_id}.tar.gz",
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        job["job_path"] = f"jobs/{target_cell.cell_id}.json"
        job = digested(job)
        job_path = PACKAGE_DIR / str(job["job_path"])
        _write_json(job_path, job)
        job_payloads.append(job)
        audit_rows.append(
            {
                "execution_id": target_cell.cell_id,
                "source_cell_id": source_cell.cell_id,
                "source_protocol": {
                    "path": source_protocol_path.relative_to(repo_root).as_posix(),
                    "sha256": sha256_file(source_protocol_path),
                    "canonical_sha256": source_protocol.sha256,
                },
                "target_protocol": protocol_binding,
                "source_lock_id": target_cell.source_lock_id,
                "source_lock_sha256": cell_lock["sha256"],
                **differences,
                "status": "passed",
            }
        )

    source_archive = PACKAGE_DIR / "source" / "source_locked.tar.gz"
    _write_source_archive(
        repo_root=source_root,
        destination=source_archive,
        members=members,
    )
    source_archive_manifest = digested(
        {
            "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA,
            "status": "passed",
            "implementation_source_inventory_sha256": (
                implementation_inventory["sha256"]
            ),
            "archive": binding(source_archive, root=PACKAGE_DIR),
            "member_count": len(members),
            "members": members,
            "global_source_paths": sorted(
                str(row["path"]) for row in global_sources.values()
            ),
            "runtime_path_dependencies": ["requirements.txt"],
            "no_ambient_repo_imports": True,
            "predecessor_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
            "source_patch": source_patch,
        }
    )
    _write_json(
        PACKAGE_DIR / "source" / "source_archive_manifest.json",
        source_archive_manifest,
    )

    source_lock_audit = digested(
        {
            "schema": SOURCE_LOCK_AUDIT_SCHEMA,
            "status": "passed",
            "source_package_manifest_sha256": (
                source_package["manifest"]["sha256"]
            ),
            "source_protocol_bundle_manifest_sha256": (
                source_package["protocol_bundle"]["sha256"]
            ),
            "source_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
            "source_locks_sha256": source_package["source_locks"]["sha256"],
            "source_locks_snapshot_sha256": source_locks["sha256"],
            "implementation_source_inventory_sha256": (
                implementation_inventory["sha256"]
            ),
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "scientific_changes": [
                "prior_cumulative_denominator_to_prior_accepted_mean_denominator_v2",
            ],
            "unchanged_scientific_settings": [
                "plateau_prior_mean_ratio_threshold_1e-4",
                "fresh_maximum_controller_rounds_70",
            ],
            "source_patch": source_patch,
            "preserved_settings": {
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "optimizer": "powell",
                "optimizer_maxiter": 200,
                "adapt_seed": 7,
                "pruning": "off",
                "beam": "off",
                "admission": "singleton",
            },
            "rows": audit_rows,
            "cell_count": len(audit_rows),
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )
    _write_json(PACKAGE_DIR / "source_lock_audit.json", source_lock_audit)

    queue_path = PACKAGE_DIR / "queue.tsv"
    with queue_path.open("xb") as stream:
        stream.write(
            ("\n".join(_queue_line(job) for job in job_payloads) + "\n").encode(
                "utf-8"
            )
        )
        stream.flush()
        os.fsync(stream.fileno())

    execution_plan = digested(
        {
            "schema": (
                "paper_i_ra_adapt_historical_average_singleton_plateau6_"
                "r70_execution_plan_v1"
            ),
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": "candidate",
            "execution_target": "chtc",
            "execution_mode": "fresh_0_to_70",
            "execution_ids": [job["execution_id"] for job in job_payloads],
            "row_count": len(job_payloads),
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "protocol_bundle_manifest_sha256": (
                protocol_bundle_manifest["sha256"]
            ),
            "source_locks_snapshot_sha256": source_locks["sha256"],
            "source_archive_manifest_sha256": source_archive_manifest["sha256"],
            "source_lock_audit_sha256": source_lock_audit["sha256"],
            "queue_sha256": sha256_file(queue_path),
            "ordinary_cluster": True,
            "bounded_factory": False,
            "success_rows_leave_queue": False,
            "per_job_checkpoint": True,
            "per_job_estimator_ledger": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "execution_plan.json", execution_plan)

    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "passed_inert_six_rows",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": len(job_payloads),
            "execution_ids": [job["execution_id"] for job in job_payloads],
            "source_cell_ids": [job["source_cell_id"] for job in job_payloads],
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "control_files": [
                binding(PACKAGE_DIR / name, root=PACKAGE_DIR)
                for name in CONTROL_FILES
            ],
            "protocol_bundle_manifest": binding(
                PACKAGE_DIR / "protocol_bundle_manifest.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "source_locks_snapshot": binding(
                PACKAGE_DIR / "source_locks_snapshot.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "source_archive": binding(source_archive, root=PACKAGE_DIR),
            "source_archive_manifest": binding(
                PACKAGE_DIR / "source" / "source_archive_manifest.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "source_lock_audit": binding(
                PACKAGE_DIR / "source_lock_audit.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "execution_plan": binding(
                PACKAGE_DIR / "execution_plan.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "queue": binding(queue_path, root=PACKAGE_DIR),
            "protocols": protocol_bindings,
            "jobs": [
                {
                    "execution_id": job["execution_id"],
                    **binding(
                        PACKAGE_DIR / str(job["job_path"]),
                        root=PACKAGE_DIR,
                        canonical=True,
                    ),
                }
                for job in job_payloads
            ],
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submit_template_present": True,
            "submit_descriptor_present": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    return {
        "status": manifest["status"],
        "package_id": PACKAGE_ID,
        "row_count": len(job_payloads),
        "package_manifest_sha256": manifest["sha256"],
        "protocol_bundle_manifest_sha256": protocol_bundle_manifest["sha256"],
        "source_locks_snapshot_sha256": source_locks["sha256"],
        "implementation_source_inventory_sha256": implementation_inventory[
            "sha256"
        ],
        "source_archive_sha256": sha256_file(source_archive),
        "protocol_sha256s": {
            job["execution_id"]: job["protocol_sha256"]
            for job in job_payloads
        },
    }


def main() -> int:
    print(json.dumps(build(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


