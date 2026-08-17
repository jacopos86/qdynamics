#!/usr/bin/env python3
"""Seal the one-cell named L=3 intermediate-weak Page-12 package."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import sys
import tarfile
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
    APPLICATION_SOURCE_LOCK_KEY,
    APPLICATION_SOURCE_SHA256,
    BACKEND_COMPILE_SCOPE,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    CONTROL_FILES,
    EXECUTION_ID,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    EXPECTED_CANDIDATE_FUNNEL,
    GENERATED_PATHS,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PHASE0_POLICY,
    PHASE0_SHORTLIST_SIZE,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REQUIRED_PHASE3_QISKIT_SOURCE_PATHS,
    RESOURCE_ENVELOPE,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_ID,
    RUN_CLASS,
    SAME_CUTOFF_REFERENCE_RECEIPT_SHA256,
    SELECTOR_COMPILE_COST_PHASE_REUSE,
    SELECTOR_COMPILE_COST_POLICY,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    SOURCE_LOCK_ID,
    SOURCE_ROUTE_CONTRACT_SHA256,
    SOURCE_ROUTE_PROFILE,
    STAGE_ID,
    TARGET_HORIZON,
    TARGET_ROUTE_CONTRACT_SHA256,
    TARGET_ROUTE_PROFILE,
    PackageContractError,
    binding,
    canonical_json_bytes,
    digested,
    repo_root_from_script,
    sha256_file,
)


REPO_ROOT = repo_root_from_script(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    BundleCellSpec,
    _bundle_protocol_materialization_authority,
    _implementation_source_inventory,
)
from pipelines.static_adapt.ra_adapt.engine import (  # noqa: E402
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.ra_adapt.l3_page12 import (  # noqa: E402
    PAPER_I_L3_PAGE12_ADAPTER_ID,
    PAPER_I_L3_PAGE12_ALGORITHM_ID,
    PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256,
    PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY,
    build_paper_i_l3_page12_problem,
    build_paper_i_l3_page12_request,
    paper_i_l3_page12_application_source_contract,
)


BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _cell() -> BundleCellSpec:
    return BundleCellSpec(
        cell_id=EXECUTION_ID,
        stage=STAGE_ID,
        regime_id="intermediate_weak",
        nph=1,
        route_id=ROUTE_ID,
        algorithm_id=ALGORITHM_ID,
        selector_family="ra_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION,
        horizon=TARGET_HORIZON,
        source_lock_id=SOURCE_LOCK_ID,
    )


def _source_locks(
    *,
    implementation: Mapping[str, Any],
    application_source: Mapping[str, Any],
) -> dict[str, Any]:
    cell_lock = digested(
        {
            "schema": "paper_i_l3_page12_cell_source_lock_v1",
            "source_lock_id": SOURCE_LOCK_ID,
            "application_source_contract_sha256": application_source["sha256"],
            "problem_request_sha256": application_source["problem_request_sha256"],
            "hamiltonian_terms_sha256": application_source[
                "hamiltonian_terms_sha256"
            ],
            "same_cutoff_exact_reference_receipt_sha256": (
                application_source["same_cutoff_exact_reference"][
                    "receipt_sha256"
                ]
            ),
            "target_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
        }
    )
    return digested(
        {
            "schema": "paper_i_l3_page12_source_locks_v1",
            "package_id": PACKAGE_ID,
            "implementation_sources": dict(implementation),
            "application_source_contract": {
                "path": "source_authority/l3_application_source_contract.json",
                "canonical_sha256": application_source["sha256"],
            },
            "cell_locks": {SOURCE_LOCK_ID: cell_lock},
            "l2_source_protocol_count": 0,
            "l2_cell_source_lock_count": 0,
        }
    )


def _source_lock_refs(
    *,
    locks: Mapping[str, Any],
    application_source: Mapping[str, Any],
) -> dict[str, str]:
    cell_lock = locks["cell_locks"][SOURCE_LOCK_ID]
    return {
        "source_locks_manifest_sha256": str(locks["sha256"]),
        "implementation_source_inventory_sha256": str(
            locks["implementation_sources"]["sha256"]
        ),
        "cell_source_lock_id": SOURCE_LOCK_ID,
        "cell_source_lock_sha256": str(cell_lock["sha256"]),
        "ed_cutoff_reference_sha256": SAME_CUTOFF_REFERENCE_RECEIPT_SHA256,
        APPLICATION_SOURCE_LOCK_KEY: str(application_source["sha256"]),
        "l3_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
    }


def _write_source_archive(source_locks: Mapping[str, Any]) -> dict[str, Any]:
    implementation = source_locks["implementation_sources"]
    rows: list[dict[str, Any]] = []
    for raw in implementation["files"]:
        relative = str(raw["path"])
        source = REPO_ROOT / relative
        rows.append(
            {
                "path": relative,
                "sha256": str(raw["sha256"]),
                "size_bytes": int(source.stat().st_size),
                "source_kind": "verified_current_implementation_inventory",
            }
        )
    rows.sort(key=lambda row: row["path"])
    required = set(REQUIRED_PHASE3_QISKIT_SOURCE_PATHS)
    observed = {row["path"] for row in rows}
    missing = sorted(required - observed)
    if missing:
        raise PackageContractError(
            "Current implementation inventory omitted required L3 route source: "
            + ", ".join(missing)
        )
    for row in rows:
        source = REPO_ROOT / row["path"]
        if (
            not source.is_file()
            or source.is_symlink()
            or sha256_file(source) != row["sha256"]
            or source.stat().st_size != row["size_bytes"]
        ):
            raise PackageContractError(f"Source member drifted: {source}")
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(
                mode="w", fileobj=gz, format=tarfile.PAX_FORMAT
            ) as archive:
                for row in rows:
                    source = REPO_ROOT / row["path"]
                    info = tarfile.TarInfo(row["path"])
                    info.size = source.stat().st_size
                    info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
    archive_binding = binding(archive_path, root=PACKAGE_DIR)
    manifest = digested(
        {
            "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA,
            "status": "passed",
            "package_id": PACKAGE_ID,
            "implementation_source_inventory_sha256": implementation["sha256"],
            "member_count": len(rows),
            "members": rows,
            "archive": archive_binding,
            "no_ambient_repo_imports": True,
        }
    )
    _write_json(PACKAGE_DIR / "source/source_archive_manifest.json", manifest)
    return manifest


def _expected_artifacts() -> dict[str, Any]:
    root = f"runs/{EXECUTION_ID}"
    suffixes = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
    }
    return {
        role: {
            "path": f"{root}/{suffix}",
            "required": True,
            "direct_file_required": True,
            "reference_receipt_required": False,
            "fulfillment_kind": "direct_execution_v1",
        }
        for role, suffix in suffixes.items()
    }


def build() -> dict[str, Any]:
    if any((PACKAGE_DIR / name).exists() for name in GENERATED_PATHS):
        raise FileExistsError("Refusing to overwrite an existing package seal.")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file():
            raise PackageContractError(f"Missing control file: {name}")
    if not (
        ALGORITHM_ID == PAPER_I_L3_PAGE12_ALGORITHM_ID
        and CANDIDATE_ADAPTER_ID == PAPER_I_L3_PAGE12_ADAPTER_ID
        and APPLICATION_SOURCE_LOCK_KEY == PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY
        and APPLICATION_SOURCE_SHA256
        == PAPER_I_L3_PAGE12_APPLICATION_SOURCE_SHA256
        and TARGET_ROUTE_CONTRACT_SHA256
        == PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError("Package constants drifted from the named L3 seam.")

    problem = build_paper_i_l3_page12_problem()
    request = build_paper_i_l3_page12_request()
    application_source = paper_i_l3_page12_application_source_contract(problem)
    _write_json(
        PACKAGE_DIR / "source_authority/l3_application_source_contract.json",
        application_source,
    )
    implementation = _implementation_source_inventory(REPO_ROOT)
    locks = _source_locks(
        implementation=implementation,
        application_source=application_source,
    )
    cell = _cell()
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(BUNDLE_ROOT / "source_locks.json", locks)
    expected = digested(
        {
            "schema": "paper_i_l3_page12_expected_artifacts_v1",
            "bundle_id": BUNDLE_ID,
            "cells": {
                EXECUTION_ID: {"expected_run_artifacts": _expected_artifacts()}
            },
        }
    )
    _write_json(BUNDLE_ROOT / "expected_artifacts.json", expected)
    bundle_manifest = digested(
        {
            "schema": "paper_i_l3_page12_bundle_manifest_v1",
            "bundle_id": BUNDLE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "cell_count": 1,
            "cells": [cell.to_dict()],
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "algorithm_id": ALGORITHM_ID,
            "target_horizon": TARGET_HORIZON,
            "source_locks_sha256": locks["sha256"],
            "expected_artifacts_sha256": expected["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    _write_json(BUNDLE_ROOT / "bundle_manifest.json", bundle_manifest)
    refs = _source_lock_refs(locks=locks, application_source=application_source)
    authority = _bundle_protocol_materialization_authority(
        cell=cell,
        bundle_id=BUNDLE_ID,
        bundle_manifest_sha256=bundle_manifest["sha256"],
        source_locks_sha256=locks["sha256"],
        source_lock_refs=refs,
        active_gradient_policy=ACTIVE_GRADIENT_POLICY,
        resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
    )
    protocol = build_resolved_ra_protocol(
        problem,
        request,
        materialization_authority=authority,
    )
    route = protocol.route_contract
    if not isinstance(route, Mapping):
        raise PackageContractError("Named L3 protocol lacks a route contract.")
    invariants = route.get("semantic_invariants")
    execution = route.get("execution_settings")
    if (
        protocol.algorithm_id != ALGORITHM_ID
        or protocol.adapter_id != CANDIDATE_ADAPTER_ID
        or protocol.horizon != TARGET_HORIZON
        or protocol.problem.num_sites != 3
        or protocol.problem.n_ph_max != 1
        or protocol.problem.n_fermions is not None
        or route.get("sha256") != TARGET_ROUTE_CONTRACT_SHA256
        or route.get("route_profile") != TARGET_ROUTE_PROFILE
        or not isinstance(invariants, Mapping)
        or not isinstance(execution, Mapping)
        or invariants.get("candidate_funnel_order") != EXPECTED_CANDIDATE_FUNNEL
        or invariants.get("phase0_active") is not True
        or invariants.get("physical_operator_lanes_active") is not False
        or invariants.get("phase_ii_compile_cost_source")
        != "backend_transpile_v1"
        or invariants.get("phase_iii_compile_cost_source")
        != "backend_transpile_v1"
        or invariants.get("selector_compile_cost_policy")
        != SELECTOR_COMPILE_COST_POLICY
        or invariants.get("selector_compile_cost_phase_reuse")
        != SELECTOR_COMPILE_COST_PHASE_REUSE
        or execution.get("phase3_backend_cost_scope") != BACKEND_COMPILE_SCOPE
        or protocol.source_locks.get(APPLICATION_SOURCE_LOCK_KEY)
        != APPLICATION_SOURCE_SHA256
    ):
        raise PackageContractError("Named L3 Page-12 protocol drifted.")
    protocol_path = BUNDLE_ROOT / "protocols" / f"{EXECUTION_ID}.json"
    _write_json(protocol_path, protocol.to_dict())
    protocol_binding = {
        "execution_id": EXECUTION_ID,
        **binding(protocol_path, root=PACKAGE_DIR, canonical=True),
    }
    job = digested(
        {
            "schema": JOB_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "execution_id": EXECUTION_ID,
            "cell_id": EXECUTION_ID,
            "regime_id": "intermediate_weak",
            "num_sites": 3,
            "nph": 1,
            "target_horizon": TARGET_HORIZON,
            "algorithm_id": ALGORITHM_ID,
            "route_id": ROUTE_ID,
            "route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "application_source_contract_sha256": APPLICATION_SOURCE_SHA256,
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "candidate_representation": CANDIDATE_REPRESENTATION,
            "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
            "selector_compile_cost_scope": BACKEND_COMPILE_SCOPE,
            "protocol_path": protocol_binding["path"],
            "protocol_file_sha256": protocol_binding["sha256"],
            "protocol_sha256": protocol.sha256,
            "bundle_manifest_sha256": bundle_manifest["sha256"],
            "source_locks_sha256": locks["sha256"],
            "implementation_source_inventory_sha256": implementation["sha256"],
            "expected_artifacts_manifest_sha256": expected["sha256"],
            "expected_run_artifacts": _expected_artifacts(),
            "resources": dict(RESOURCE_ENVELOPE),
            "expected_output_archive": f"{EXECUTION_ID}.tar.gz",
            "fresh_start_contract": {
                "kind": "fresh_start",
                "source_checkpoint": None,
                "resume_archive": None,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    job_path = PACKAGE_DIR / "jobs" / f"{EXECUTION_ID}.json"
    _write_json(job_path, job)
    validation = digested(
        {
            "schema": "paper_i_l3_page12_validation_v1",
            "status": "passed",
            "bundle_id": BUNDLE_ID,
            "protocol_count": 1,
            "route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
            "application_source_contract_sha256": APPLICATION_SOURCE_SHA256,
            "problem": {
                "num_sites": 3,
                "n_ph_max": 1,
                "num_qubits": 9,
                "num_particles": [2, 1],
            },
            "qiskit_cost_phases": ["phase_ii", "phase_iii"],
            "lane_shortlisting_disabled": True,
            "target_horizon": TARGET_HORIZON,
        }
    )
    _write_json(BUNDLE_ROOT / "validation_report.json", validation)
    source_manifest = _write_source_archive(locks)
    queue_path = PACKAGE_DIR / "queue.tsv"
    with queue_path.open("xb") as stream:
        stream.write(
            (
                "\t".join(
                    (
                        EXECUTION_ID,
                        f"jobs/{EXECUTION_ID}.json",
                        protocol_binding["path"],
                        sha256_file(job_path),
                        str(RESOURCE_ENVELOPE["request_cpus"]),
                        str(RESOURCE_ENVELOPE["request_memory_mb"]),
                        str(RESOURCE_ENVELOPE["request_disk_mb"]),
                        str(RESOURCE_ENVELOPE["max_runtime_seconds"]),
                    )
                )
                + "\n"
            ).encode("utf-8")
        )
    plan = digested(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 1,
            "execution_ids": [EXECUTION_ID],
            "route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
            "application_source_contract_sha256": APPLICATION_SOURCE_SHA256,
            "source_archive_sha256": source_manifest["archive"]["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "execution_plan.json", plan)
    audit = digested(
        {
            "schema": "paper_i_l3_page12_source_lock_audit_v1",
            "status": "passed",
            "source_route_profile": SOURCE_ROUTE_PROFILE,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "target_route_profile": TARGET_ROUTE_PROFILE,
            "target_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
            "application_source_contract_sha256": APPLICATION_SOURCE_SHA256,
            "implementation_source_inventory_sha256": implementation["sha256"],
            "l2_source_protocol_count": 0,
            "l2_cell_source_lock_count": 0,
            "scientific_result_anchor_claimed": False,
        }
    )
    _write_json(PACKAGE_DIR / "source_lock_audit.json", audit)
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "passed_inert_single_cell",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "row_count": 1,
            "execution_ids": [EXECUTION_ID],
            "child_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "application_source_contract_sha256": APPLICATION_SOURCE_SHA256,
            "implementation_source_inventory_sha256": implementation["sha256"],
            "bundle_manifest": binding(
                BUNDLE_ROOT / "bundle_manifest.json", root=PACKAGE_DIR, canonical=True
            ),
            "bundle_expected_artifacts": binding(
                BUNDLE_ROOT / "expected_artifacts.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_source_locks": binding(
                BUNDLE_ROOT / "source_locks.json", root=PACKAGE_DIR, canonical=True
            ),
            "bundle_validation_report": binding(
                BUNDLE_ROOT / "validation_report.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "application_source_contract": binding(
                PACKAGE_DIR / "source_authority/l3_application_source_contract.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "protocols": [protocol_binding],
            "jobs": [
                {
                    "execution_id": EXECUTION_ID,
                    **binding(job_path, root=PACKAGE_DIR, canonical=True),
                }
            ],
            "source_archive": source_manifest["archive"],
            "source_archive_manifest": binding(
                PACKAGE_DIR / "source/source_archive_manifest.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "source_archive_manifest_sha256": source_manifest["sha256"],
            "queue": binding(queue_path, root=PACKAGE_DIR),
            "execution_plan": binding(
                PACKAGE_DIR / "execution_plan.json", root=PACKAGE_DIR, canonical=True
            ),
            "source_lock_audit": binding(
                PACKAGE_DIR / "source_lock_audit.json", root=PACKAGE_DIR, canonical=True
            ),
            "control_files": [
                binding(PACKAGE_DIR / name, root=PACKAGE_DIR)
                for name in CONTROL_FILES
            ],
            "required_route_source_paths": list(REQUIRED_PHASE3_QISKIT_SOURCE_PATHS),
            "remote_image_path": REMOTE_IMAGE_PATH,
            "remote_image_sha256": REMOTE_IMAGE_SHA256,
            "target_horizon": TARGET_HORIZON,
            "activation_artifacts_present": False,
            "authorizations_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submit_descriptor_present": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    return {
        "status": "passed_inert_single_cell",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
        "application_source_contract_sha256": APPLICATION_SOURCE_SHA256,
        "source_archive_sha256": source_manifest["archive"]["sha256"],
        "row_count": 1,
    }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
