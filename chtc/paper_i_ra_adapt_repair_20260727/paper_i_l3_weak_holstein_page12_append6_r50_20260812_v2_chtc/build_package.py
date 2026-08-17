#!/usr/bin/env python3
"""Seal the inert matched six-cell L=3 weak-Holstein package."""
from __future__ import annotations

import gzip
import os
from pathlib import Path
import sys
import tarfile
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import *  # noqa: E402,F403

REPO_ROOT = repo_root_from_script(__file__)
sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.ra_adapt.append import (  # noqa: E402
    APPEND_ADAPT_ALGORITHM_ID,
    build_resolved_append_protocol,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    BundleCellSpec,
    _bundle_protocol_materialization_authority,
    _implementation_source_inventory,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    ACTIVE_GRADIENT_MEASURED,
    ACTIVE_GRADIENT_STATIONARY,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    AppendAdaptRequest,
)
from pipelines.static_adapt.ra_adapt.engine import build_resolved_ra_protocol  # noqa: E402
from pipelines.static_adapt.ra_adapt.l3_page12 import (  # noqa: E402
    PAPER_I_L3_PAGE12_ADAPTER_ID,
    PAPER_I_L3_PAGE12_ALGORITHM_ID,
    PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256,
    PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY,
    PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256,
    PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter,
    build_paper_i_l3_page12_problem,
    build_paper_i_l3_page12_request,
    paper_i_l3_page12_application_source_contract,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    CheckpointObservation,
    EstimatorLedgerObservation,
    FreshStart,
    SRExecutionPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)

BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush(); os.fsync(stream.fileno())

def _cells() -> tuple[BundleCellSpec, ...]:
    rows = []
    for regime in REGIMES:
        for method in METHODS:
            rows.append(BundleCellSpec(
                cell_id=execution_id(regime, method),
                stage=f"l3_{regime}_{method}_candidate",
                regime_id=regime,
                nph=3,
                route_id=RA_ROUTE_ID if method == "ra_page12" else APPEND_ROUTE_ID,
                algorithm_id=RA_ALGORITHM_ID if method == "ra_page12" else APPEND_ALGORITHM_ID,
                selector_family="ra_adapt" if method == "ra_page12" else "append_adapt",
                candidate_representation=CANDIDATE_REPRESENTATION,
                horizon=TARGET_HORIZON,
                source_lock_id=source_lock_id(regime),
            ))
    return tuple(rows)

def _expected_artifacts(execution: str) -> dict[str, Any]:
    root = f"runs/{execution}"
    suffixes = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
    }
    return {role: {"path": f"{root}/{suffix}", "required": True,
                   "direct_file_required": True, "reference_receipt_required": False,
                   "fulfillment_kind": "direct_execution_v1"}
            for role, suffix in suffixes.items()}

def _source_locks(implementation: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    cell_locks = {}
    for regime, source in sources.items():
        lock_id = source_lock_id(regime)
        cell_locks[lock_id] = digested({
            "schema": "paper_i_l3_weak_holstein_cell_source_lock_v1",
            "source_lock_id": lock_id,
            "regime_id": regime,
            "application_source_contract_sha256": source["sha256"],
            "problem_request_sha256": source["problem_request_sha256"],
            "hamiltonian_terms_sha256": source["hamiltonian_terms_sha256"],
            "same_cutoff_exact_reference_receipt_sha256": source["same_cutoff_exact_reference"]["receipt_sha256"],
            "ra_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
        })
    return digested({
        "schema": "paper_i_l3_weak_holstein_source_locks_v1",
        "package_id": PACKAGE_ID,
        "implementation_sources": dict(implementation),
        "application_source_contracts": {
            regime: {"path": f"source_authority/{regime}_application_source_contract.json",
                     "canonical_sha256": source["sha256"]}
            for regime, source in sources.items()
        },
        "cell_locks": cell_locks,
        "append_runtime_hash_dependencies": list(APPEND_RUNTIME_SOURCE_DEPENDENCIES),
        "l2_source_protocol_count": 0,
        "l2_cell_source_lock_count": 0,
    })

def _refs(locks: Mapping[str, Any], source: Mapping[str, Any], regime: str) -> dict[str, str]:
    lock_id = source_lock_id(regime); lock = locks["cell_locks"][lock_id]
    return {
        "source_locks_manifest_sha256": str(locks["sha256"]),
        "implementation_source_inventory_sha256": str(locks["implementation_sources"]["sha256"]),
        "cell_source_lock_id": lock_id,
        "cell_source_lock_sha256": str(lock["sha256"]),
        "ed_cutoff_reference_sha256": str(source["same_cutoff_exact_reference"]["receipt_sha256"]),
        APPLICATION_SOURCE_LOCK_KEY: str(source["sha256"]),
        "l3_route_contract_sha256": TARGET_ROUTE_CONTRACT_SHA256,
    }

def _source_archive(locks: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for raw in locks["implementation_sources"]["files"]:
        source = REPO_ROOT / str(raw["path"])
        rows.append({"path": str(raw["path"]), "sha256": str(raw["sha256"]),
                     "size_bytes": source.stat().st_size,
                     "source_kind": "verified_current_implementation_inventory"})
    by_path = {row["path"]: row for row in rows}
    for raw in APPEND_RUNTIME_SOURCE_DEPENDENCIES:
        relative = str(raw["path"]); source = REPO_ROOT / relative
        candidate = {"path": relative, "sha256": str(raw["sha256"]),
                     "size_bytes": int(raw["size_bytes"]),
                     "source_kind": "append_runtime_hash_dependency"}
        if relative in by_path and by_path[relative]["sha256"] != candidate["sha256"]:
            raise PackageContractError("Append runtime dependency conflicts with implementation inventory.")
        by_path[relative] = candidate
    rows = sorted(by_path.values(), key=lambda row: row["path"])
    missing = sorted(set(REQUIRED_ROUTE_SOURCE_PATHS) - set(by_path))
    if missing: raise PackageContractError("Source archive omitted required route source: " + ", ".join(missing))
    for row in rows:
        source = REPO_ROOT / row["path"]
        if (not source.is_file() or source.is_symlink() or sha256_file(source) != row["sha256"]
                or source.stat().st_size != row["size_bytes"]):
            raise PackageContractError(f"Source member drifted: {source}")
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    source = REPO_ROOT / row["path"]
                    info = tarfile.TarInfo(row["path"]); info.size = source.stat().st_size
                    info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                    info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
                    with source.open("rb") as stream: archive.addfile(info, stream)
    manifest = digested({
        "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA, "status": "passed",
        "package_id": PACKAGE_ID,
        "implementation_source_inventory_sha256": locks["implementation_sources"]["sha256"],
        "member_count": len(rows), "members": rows,
        "archive": binding(archive_path, root=PACKAGE_DIR), "no_ambient_repo_imports": True,
    })
    _write_json(PACKAGE_DIR / "source/source_archive_manifest.json", manifest)
    return manifest

def _append_request(_execution: str) -> AppendAdaptRequest:
    return AppendAdaptRequest(
        adapter=PaperIL3Page12GlobalSingletonGradientPhase0CandidateAdapter(),
        execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=50), resume=FreshStart()),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(path=Path("checkpoints/current.json"),
                                             every_controller_rounds=1, keep_history_tail=100),
            estimator_ledger=EstimatorLedgerObservation(path=Path("result/estimator_ledger.json")),
            resource_rounds=(50,),
        ),
    )

def build() -> dict[str, Any]:
    if any((PACKAGE_DIR / name).exists() for name in GENERATED_PATHS):
        raise FileExistsError("Refusing to overwrite an existing package seal.")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file(): raise PackageContractError(f"Missing control file: {name}")
    if (RA_ALGORITHM_ID != PAPER_I_L3_PAGE12_ALGORITHM_ID or APPEND_ALGORITHM_ID != APPEND_ADAPT_ALGORITHM_ID
            or CANDIDATE_ADAPTER_ID != PAPER_I_L3_PAGE12_ADAPTER_ID
            or APPLICATION_SOURCE_LOCK_KEY != PAPER_I_L3_PAGE12_SOURCE_LOCK_KEY
            or TARGET_ROUTE_CONTRACT_SHA256 != PAPER_I_L3_PAGE12_ROUTE_CONTRACT_SHA256):
        raise PackageContractError("Package constants drifted from named L3 seams.")
    sources = {}; problems = {}
    for regime in REGIMES:
        problem = build_paper_i_l3_page12_problem(regime, nph=3)
        source = paper_i_l3_page12_application_source_contract(problem)
        if source["sha256"] != PAPER_I_L3_PAGE12_WEAK_SECTOR_NPH3_APPLICATION_SOURCE_SHA256[regime]:
            raise PackageContractError("Application source lock drifted.")
        problems[regime] = problem; sources[regime] = source
        _write_json(PACKAGE_DIR / f"source_authority/{regime}_application_source_contract.json", source)
    implementation = _implementation_source_inventory(REPO_ROOT)
    locks = _source_locks(implementation, sources)
    cells = _cells(); BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(BUNDLE_ROOT / "source_locks.json", locks)
    expected = digested({"schema": "paper_i_l3_weak_holstein_expected_artifacts_v1",
                         "bundle_id": BUNDLE_ID,
                         "cells": {cell.cell_id: {"expected_run_artifacts": _expected_artifacts(cell.cell_id)} for cell in cells}})
    _write_json(BUNDLE_ROOT / "expected_artifacts.json", expected)
    bundle_manifest = digested({
        "schema": "paper_i_l3_weak_holstein_bundle_manifest_v1", "bundle_id": BUNDLE_ID,
        "campaign_id": CAMPAIGN_ID, "run_class": RUN_CLASS, "cell_count": len(cells),
        "cells": [cell.to_dict() for cell in cells], "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
        "target_horizon": TARGET_HORIZON, "source_locks_sha256": locks["sha256"],
        "expected_artifacts_sha256": expected["sha256"], "execution_authorized": False, "submitted": False,
    })
    _write_json(BUNDLE_ROOT / "bundle_manifest.json", bundle_manifest)
    protocols = []; jobs = []; queue_rows = []
    for cell in cells:
        method = "ra_page12" if cell.selector_family == "ra_adapt" else "append_adapt"
        source = sources[cell.regime_id]; refs = _refs(locks, source, cell.regime_id)
        policy = ACTIVE_GRADIENT_RA if method == "ra_page12" else ACTIVE_GRADIENT_APPEND
        authority = _bundle_protocol_materialization_authority(
            cell=cell, bundle_id=BUNDLE_ID, bundle_manifest_sha256=bundle_manifest["sha256"],
            source_locks_sha256=locks["sha256"], source_lock_refs=refs,
            active_gradient_policy=policy, resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
        )
        protocol = (build_resolved_ra_protocol(problems[cell.regime_id], build_paper_i_l3_page12_request(),
                                               materialization_authority=authority)
                    if method == "ra_page12" else
                    build_resolved_append_protocol(problems[cell.regime_id], _append_request(cell.cell_id),
                                                   materialization_authority=authority))
        if method == "ra_page12":
            if (protocol.route_contract["sha256"] != TARGET_ROUTE_CONTRACT_SHA256
                    or protocol.request.method.pruning.kind != "off" or protocol.request.method.beam.kind != "off"):
                raise PackageContractError("L3 Page12 RA route drifted.")
            route_sha = TARGET_ROUTE_CONTRACT_SHA256; entrypoint = "run_ra_adapt"
        else:
            if (protocol.selector_identity != APPEND_CONVENTIONAL_SELECTOR_ID
                    or protocol.lineage_authority.get("ra_staged_funnel_invoked") is not False
                    or protocol.algorithm_id != APPEND_ALGORITHM_ID):
                raise PackageContractError("Conventional Append facade drifted.")
            route_sha = digested({"schema": "paper_i_l3_conventional_append_route_v1",
                                  "selector_identity": protocol.selector_identity,
                                  "selector_scope": protocol.selector_scope,
                                  "ra_staged_funnel_invoked": False})["sha256"]
            entrypoint = "run_append_adapt"
        if (protocol.problem.num_sites != 3 or protocol.problem.n_ph_max != 3 or protocol.horizon != 50
                or protocol.optimizer != "powell" or protocol.optimizer_maxiter != 200 or protocol.seeds["adapt"] != 7):
            raise PackageContractError("Matched scientific settings drifted.")
        protocol_path = BUNDLE_ROOT / "protocols" / f"{cell.cell_id}.json"; _write_json(protocol_path, protocol.to_dict())
        protocol_binding = {"execution_id": cell.cell_id, **binding(protocol_path, root=PACKAGE_DIR, canonical=True)}
        resources = RESOURCE_ENVELOPES[method]
        job = digested({
            "schema": JOB_SCHEMA, "package_id": PACKAGE_ID, "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID, "execution_id": cell.cell_id, "cell_id": cell.cell_id,
            "regime_id": cell.regime_id, "method": method, "execution_entrypoint": entrypoint,
            "num_sites": 3, "nph": 3, "target_horizon": 50, "algorithm_id": cell.algorithm_id,
            "route_id": cell.route_id, "route_contract_sha256": route_sha,
            "application_source_contract_sha256": source["sha256"],
            "active_gradient_policy": policy, "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "candidate_representation": CANDIDATE_REPRESENTATION, "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
            "protocol_path": protocol_binding["path"], "protocol_file_sha256": protocol_binding["sha256"],
            "protocol_sha256": protocol.sha256, "bundle_manifest_sha256": bundle_manifest["sha256"],
            "source_locks_sha256": locks["sha256"], "implementation_source_inventory_sha256": implementation["sha256"],
            "expected_artifacts_manifest_sha256": expected["sha256"],
            "expected_run_artifacts": _expected_artifacts(cell.cell_id), "resources": dict(resources),
            "expected_output_archive": f"{cell.cell_id}.tar.gz",
            "fresh_start_contract": {"kind": "fresh_start", "source_checkpoint": None, "resume_archive": None},
            "execution_authorized": False, "submission_authorized": False, "submitted": False,
        })
        job_path = PACKAGE_DIR / "jobs" / f"{cell.cell_id}.json"; _write_json(job_path, job)
        protocols.append(protocol_binding); jobs.append({"execution_id": cell.cell_id, **binding(job_path, root=PACKAGE_DIR, canonical=True)})
        queue_rows.append("\t".join((cell.cell_id, f"jobs/{cell.cell_id}.json", protocol_binding["path"],
                                     sha256_file(job_path), str(resources["request_cpus"]),
                                     str(resources["request_memory_mb"]), str(resources["request_disk_mb"]),
                                     str(resources["max_runtime_seconds"]))))
    validation = digested({
        "schema": "paper_i_l3_weak_holstein_matched_validation_v1", "status": "passed",
        "bundle_id": BUNDLE_ID, "protocol_count": 6, "regimes": list(REGIMES),
        "method_counts": {"ra_page12": 3, "append_adapt": 3}, "distinct_execution_facades": True,
        "ra_pruning": "off", "ra_beam": "off", "target_horizon": 50,
    }); _write_json(BUNDLE_ROOT / "validation_report.json", validation)
    source_manifest = _source_archive(locks)
    queue_path = PACKAGE_DIR / "queue.tsv"
    with queue_path.open("xb") as stream: stream.write(("\n".join(queue_rows) + "\n").encode())
    plan = digested({"schema": EXECUTION_PLAN_SCHEMA, "package_id": PACKAGE_ID, "campaign_id": CAMPAIGN_ID,
                     "row_count": 6, "execution_ids": list(expected_execution_ids()),
                     "execution_entrypoint_counts": {"run_ra_adapt": 3, "run_append_adapt": 3},
                     "source_archive_sha256": source_manifest["archive"]["sha256"],
                     "execution_authorized": False, "submitted": False})
    _write_json(PACKAGE_DIR / "execution_plan.json", plan)
    audit = digested({"schema": "paper_i_l3_weak_holstein_source_lock_audit_v1", "status": "passed",
                      "implementation_source_inventory_sha256": implementation["sha256"],
                      "explicit_append_runtime_dependency_count": len(APPEND_RUNTIME_SOURCE_DEPENDENCIES),
                      "l2_source_protocol_count": 0, "scientific_result_anchor_claimed": False})
    _write_json(PACKAGE_DIR / "source_lock_audit.json", audit)
    manifest = digested({
        "schema": PACKAGE_MANIFEST_SCHEMA, "status": "passed_inert_matched_six_cell",
        "package_id": PACKAGE_ID, "campaign_id": CAMPAIGN_ID, "bundle_id": BUNDLE_ID,
        "run_class": RUN_CLASS, "execution_target": EXECUTION_TARGET, "row_count": 6,
        "execution_ids": list(expected_execution_ids()), "execution_entrypoint_counts": {"run_ra_adapt": 3, "run_append_adapt": 3},
        "bundle_manifest": binding(BUNDLE_ROOT / "bundle_manifest.json", root=PACKAGE_DIR, canonical=True),
        "bundle_expected_artifacts": binding(BUNDLE_ROOT / "expected_artifacts.json", root=PACKAGE_DIR, canonical=True),
        "bundle_source_locks": binding(BUNDLE_ROOT / "source_locks.json", root=PACKAGE_DIR, canonical=True),
        "bundle_validation_report": binding(BUNDLE_ROOT / "validation_report.json", root=PACKAGE_DIR, canonical=True),
        "application_source_contracts": [binding(PACKAGE_DIR / f"source_authority/{r}_application_source_contract.json", root=PACKAGE_DIR, canonical=True) for r in REGIMES],
        "protocols": protocols, "jobs": jobs, "source_archive": source_manifest["archive"],
        "source_archive_manifest": binding(PACKAGE_DIR / "source/source_archive_manifest.json", root=PACKAGE_DIR, canonical=True),
        "source_archive_manifest_sha256": source_manifest["sha256"], "queue": binding(queue_path, root=PACKAGE_DIR),
        "execution_plan": binding(PACKAGE_DIR / "execution_plan.json", root=PACKAGE_DIR, canonical=True),
        "source_lock_audit": binding(PACKAGE_DIR / "source_lock_audit.json", root=PACKAGE_DIR, canonical=True),
        "control_files": [binding(PACKAGE_DIR / name, root=PACKAGE_DIR) for name in CONTROL_FILES],
        "required_route_source_paths": list(REQUIRED_ROUTE_SOURCE_PATHS),
        "remote_image_path": REMOTE_IMAGE_PATH, "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "target_horizon": 50, "activation_artifacts_present": False, "authorizations_present": False,
        "execution_authorized": False, "submission_authorized": False, "submission_ready": False,
        "submit_descriptor_present": False, "submitted": False, "remote_stage": False, "condor_submit": False,
    }); _write_json(PACKAGE_DIR / "package_manifest.json", manifest)
    return {"status": manifest["status"], "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"], "source_archive_sha256": source_manifest["archive"]["sha256"],
            "row_count": 6}

if __name__ == "__main__":
    try: print(canonical_json_bytes(build()).decode())
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
