#!/usr/bin/env python3
"""Seal six fresh Page-13 macro beam/metric-pruning cells through round 20."""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
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
    BASELINE_PACKAGE,
    BASELINE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    BASELINE_PACKAGE_MANIFEST_FILE_SHA256,
    BEAM_CHILDREN_PER_PARENT,
    BEAM_LIVE_BRANCHES,
    BEAM_MAXIMUM_CHILDREN_PER_ROUND,
    BEAM_S_ALG_WEIGHT,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    CONTROL_FILES,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    EXPECTED_CANDIDATE_FUNNEL,
    GENERATED_PATHS,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PHASE0_POLICY,
    PHASE0_SHORTLIST_SIZE,
    PRUNING_POLICY,
    REGIME_ROWS,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REQUIRED_ROUTE_SOURCE_PATHS,
    RESOURCE_ENVELOPES,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_ID,
    RUN_CLASS,
    STRUCTURAL_PROXY_COST_SOURCE,
    STRUCTURAL_PROXY_MODE,
    SOURCE_ROUTE_CONTRACT_SHA256,
    SOURCE_ROUTE_PROFILE,
    SOURCE_BUNDLE_ID,
    SOURCE_FINAL_RECEIPT_CANONICAL_SHA256,
    SOURCE_FINAL_RECEIPT_FILE_SHA256,
    SOURCE_LOCKS_CANONICAL_SHA256,
    SOURCE_LOCKS_FILE_SHA256,
    SOURCE_MATERIALIZATION,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    STAGE_ID,
    TARGET_ROUTE_PROFILE,
    TARGET_PARENT_ROUTE_PROFILE,
    TARGET_PARENT_ROUTE_CONTRACT_SHA256,
    TARGET_ROUTE_CONTRACT_SHA256,
    WEAK_HORIZON,
    STRONG_HORIZON,
    PackageContractError,
    binding,
    canonical_json_bytes,
    digested,
    execution_id,
    repo_root_from_script,
    sha256_file,
    source_lock_id,
    verify_self_digest,
)


REPO_ROOT = repo_root_from_script(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.contracts.problem import ProblemRequest  # noqa: E402
from pipelines.static_adapt.builders.problem_registry import (  # noqa: E402
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    BundleCellSpec,
    _bundle_protocol_materialization_authority,
    _implementation_source_inventory,
    _source_lock_refs,
)
from pipelines.static_adapt.ra_adapt.adapters import (  # noqa: E402
    MacroGradientPhase0CandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    canonical_sha256,
    resolved_ra_adapt_protocol_from_mapping,
)
from pipelines.static_adapt.ra_adapt.engine import (  # noqa: E402
    build_resolved_ra_protocol,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    ForkLocalBeam,
    MetricPruning,
)


BASELINE_PACKAGE_ROOT = REPO_ROOT / BASELINE_PACKAGE
SOURCE_MATERIALIZATION_ROOT = REPO_ROOT / SOURCE_MATERIALIZATION
SOURCE_BUNDLE_ROOT = SOURCE_MATERIALIZATION_ROOT / SOURCE_BUNDLE_ID
BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID


def _target_adapter() -> Any:
    return MacroGradientPhase0CandidateAdapter()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PackageContractError(f"Expected JSON object: {path}")
    return value


def _problem_from_protocol(protocol: Any) -> Any:
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
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def _cells() -> tuple[BundleCellSpec, ...]:
    return tuple(
        BundleCellSpec(
            cell_id=execution_id(regime, nph),
            stage=STAGE_ID,
            regime_id=regime,
            nph=nph,
            route_id=ROUTE_ID,
            algorithm_id=ALGORITHM_ID,
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION,
            horizon=horizon,
            source_lock_id=source_lock_id(regime, nph),
        )
        for regime, nph, horizon in REGIME_ROWS
    )


def _source_protocol_path(cell: BundleCellSpec) -> Path:
    return (
        SOURCE_BUNDLE_ROOT
        / "protocols"
        / f"core__{cell.regime_id}__nph{cell.nph}__ra_macro_plateau.json"
    )


def _source_protocol(cell: BundleCellSpec) -> Any:
    source_path = _source_protocol_path(cell)
    payload = _load_json(source_path)
    if verify_self_digest(payload, label=source_path.name) != payload["sha256"]:
        raise PackageContractError("Source protocol digest drifted.")
    if (
        payload.get("candidate_representation") != CANDIDATE_REPRESENTATION
        or payload.get("source_locks", {}).get("cell_source_lock_id")
        != cell.source_lock_id
    ):
        raise PackageContractError("v13 macro source protocol drifted.")
    return resolved_ra_adapt_protocol_from_mapping(payload)


def _source_locks(cells: tuple[BundleCellSpec, ...]) -> dict[str, Any]:
    old_path = SOURCE_BUNDLE_ROOT / "source_locks.json"
    old = _load_json(old_path)
    if (
        sha256_file(old_path) != SOURCE_LOCKS_FILE_SHA256
        or verify_self_digest(old, label="v13 macro source locks")
        != SOURCE_LOCKS_CANONICAL_SHA256
    ):
        raise PackageContractError("v13 macro source locks drifted.")
    implementation = _implementation_source_inventory(REPO_ROOT)
    payload = {key: value for key, value in old.items() if key != "sha256"}
    payload["implementation_sources"] = implementation
    payload["cell_locks"] = {
        cell.source_lock_id: old["cell_locks"][cell.source_lock_id]
        for cell in cells
    }
    payload["required_cell_lock_count"] = len(cells)
    payload["derivation_overlay"] = {
        "schema": "macro_gradient_phase0_proxy_no_lanes_source_overlay_v2",
        "parent_source_locks_sha256": old["sha256"],
        "algorithm_id": ALGORITHM_ID,
        "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
        "target_horizon": WEAK_HORIZON,
        "plateau_prior_mean_decrease_ratio_threshold": 1.0e-4,
        "phase0_policy": PHASE0_POLICY,
        "phase0_shortlist_size": PHASE0_SHORTLIST_SIZE,
        "phase0_estimator_components": ["N_grad"],
        "phase0_fubini_metric_active": False,
        "phase0_resource_cost_active": False,
        "phase0_compile_cost_active": False,
        "candidate_representation": CANDIDATE_REPRESENTATION,
        "selector_qiskit_compile_cost_active": False,
        "physical_operator_lanes_active": False,
        "structural_proxy_mode": STRUCTURAL_PROXY_MODE,
        "structural_proxy_cost_source": STRUCTURAL_PROXY_COST_SOURCE,
        "changed_fields": [
            "method.beam",
            "method.pruning",
            "execution.stop.maximum_controller_rounds",
            "scheduler.resources",
        ],
        "beam": {
            "kind": "fork_local",
            "live_parent_branches": BEAM_LIVE_BRANCHES,
            "admission_children_per_parent": BEAM_CHILDREN_PER_PARENT,
            "maximum_admission_children_per_round": (
                BEAM_MAXIMUM_CHILDREN_PER_ROUND
            ),
            "s_alg_weight": BEAM_S_ALG_WEIGHT,
        },
        "pruning": PRUNING_POLICY,
        "scientific_result_anchor_claimed": False,
    }
    locks = digested(payload)
    for cell in cells:
        if cell.source_lock_id not in locks.get("cell_locks", {}):
            raise PackageContractError(
                f"Missing inherited source lock for {cell.cell_id}."
            )
    return locks


def _write_source_archive(source_locks: Mapping[str, Any]) -> dict[str, Any]:
    implementation = source_locks["implementation_sources"]
    members: dict[str, dict[str, Any]] = {}
    for row in implementation["files"]:
        source_path = REPO_ROOT / str(row["path"])
        members[str(row["path"])] = {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(source_path.stat().st_size),
            "source_kind": "verified_implementation_inventory",
        }
    for source_id, row in source_locks["global_sources"].items():
        relative = str(row["path"])
        observed = {
            "path": relative,
            "sha256": str(row["sha256"]),
            "size_bytes": int((REPO_ROOT / relative).stat().st_size),
            "source_kind": f"verified_global_source:{source_id}",
        }
        previous = members.get(relative)
        if previous is not None and previous["sha256"] != observed["sha256"]:
            raise PackageContractError(f"Source-lock collision: {relative}")
        members[relative] = observed
    rows = [members[key] for key in sorted(members)]
    missing_required = sorted(
        set(REQUIRED_ROUTE_SOURCE_PATHS).difference(members)
    )
    if missing_required:
        raise PackageContractError(
            "Required route sources escaped the implementation inventory: "
            + ", ".join(missing_required)
        )
    for row in rows:
        path = REPO_ROOT / row["path"]
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["size_bytes"]
        ):
            raise PackageContractError(f"Source member drifted: {path}")
    archive_path = PACKAGE_DIR / "source/source_locked.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=False)
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as tar:
                for row in rows:
                    source = REPO_ROOT / row["path"]
                    info = tarfile.TarInfo(row["path"])
                    info.size = source.stat().st_size
                    info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)
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


def _expected_artifacts(cell_id: str) -> dict[str, Any]:
    root = f"runs/{cell_id}"
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
    baseline_manifest_path = BASELINE_PACKAGE_ROOT / "package_manifest.json"
    baseline_manifest = _load_json(baseline_manifest_path)
    if (
        sha256_file(baseline_manifest_path)
        != BASELINE_PACKAGE_MANIFEST_FILE_SHA256
        or verify_self_digest(
            baseline_manifest,
            label="Page-11 baseline package manifest",
        )
        != BASELINE_PACKAGE_MANIFEST_CANONICAL_SHA256
    ):
        raise PackageContractError("Baseline staged package manifest drifted.")
    source_receipt_path = SOURCE_MATERIALIZATION_ROOT / "final_publication_receipt.json"
    source_receipt = _load_json(source_receipt_path)
    if (
        sha256_file(source_receipt_path) != SOURCE_FINAL_RECEIPT_FILE_SHA256
        or verify_self_digest(source_receipt, label="v13 final publication receipt")
        != SOURCE_FINAL_RECEIPT_CANONICAL_SHA256
    ):
        raise PackageContractError("v13 macro publication authority drifted.")
    cells = _cells()
    locks = _source_locks(cells)
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(BUNDLE_ROOT / "source_locks.json", locks)
    expected = digested(
        {
            "schema": "macro_gradient_phase0_proxy_no_lanes_expected_artifacts_v2",
            "bundle_id": BUNDLE_ID,
            "cells": {
                cell.cell_id: {
                    "expected_run_artifacts": _expected_artifacts(cell.cell_id)
                }
                for cell in cells
            },
        }
    )
    _write_json(BUNDLE_ROOT / "expected_artifacts.json", expected)
    bundle_manifest = digested(
        {
            "schema": "macro_gradient_phase0_proxy_no_lanes_bundle_manifest_v2",
            "bundle_id": BUNDLE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "cell_count": len(cells),
            "cells": [cell.to_dict() for cell in cells],
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "algorithm_id": ALGORITHM_ID,
            "target_horizon": WEAK_HORIZON,
            "source_locks_sha256": locks["sha256"],
            "expected_artifacts_sha256": expected["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    _write_json(BUNDLE_ROOT / "bundle_manifest.json", bundle_manifest)

    protocol_bindings: list[dict[str, Any]] = []
    source_protocol_bindings: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    route_digests: set[str] = set()
    for cell in cells:
        source = _source_protocol(cell)
        source_protocol_bindings.append(
            {
                "execution_id": cell.cell_id,
                **binding(
                    _source_protocol_path(cell),
                    root=REPO_ROOT,
                    canonical=True,
                ),
            }
        )
        request = replace(
            source.request,
            adapter=_target_adapter(),
            method=replace(
                source.request.method,
                pruning=MetricPruning(),
                beam=ForkLocalBeam(
                    live_parent_branches=BEAM_LIVE_BRANCHES,
                    admission_children_per_parent=BEAM_CHILDREN_PER_PARENT,
                    maximum_admission_children_per_round=(
                        BEAM_MAXIMUM_CHILDREN_PER_ROUND
                    ),
                    s_alg_weight=BEAM_S_ALG_WEIGHT,
                ),
            ),
            execution=replace(
                source.request.execution,
                stop=replace(
                    source.request.execution.stop,
                    maximum_controller_rounds=cell.horizon,
                ),
            ),
        )
        refs = _source_lock_refs(locks, cell=cell)
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
            _problem_from_protocol(source),
            request,
            materialization_authority=authority,
        )
        route = protocol.route_contract
        if not isinstance(route, Mapping):
            raise PackageContractError("Target protocol lacks route contract.")
        execution = route["execution_settings"]
        invariants = route["semantic_invariants"]
        if (
            protocol.algorithm_id != ALGORITHM_ID
            or protocol.horizon != cell.horizon
            or route.get("route_profile") != TARGET_ROUTE_PROFILE
            or route["lineage_authority"].get("parent_contract_sha256")
            != TARGET_PARENT_ROUTE_CONTRACT_SHA256
            or route["lineage_authority"].get("parent_route_profile")
            != TARGET_PARENT_ROUTE_PROFILE
            or route.get("sha256") != TARGET_ROUTE_CONTRACT_SHA256
            or protocol.candidate_representation != CANDIDATE_REPRESENTATION
            or protocol.request.adapter.adapter_id != CANDIDATE_ADAPTER_ID
            or execution.get("phase3_backend_cost_mode")
            != STRUCTURAL_PROXY_MODE
            or "phase3_backend_cost_scope" in execution
            or execution.get("phase3_hardware_cost_normalization_mode")
            != "family_robust_symmetric_arctan_v1"
            or execution.get("static_lane_route") != "global_single_population"
            or "physical_lane_shortlist_aggressiveness" in execution
            or execution.get("adapt_inner_optimizer") != "POWELL"
            or execution.get("adapt_maxiter") != 200
            or execution.get("adapt_seed") != 7
            or execution.get("adapt_pool") != "full_meta"
            or execution.get("adapt_pool_class_filter_json") is not None
            or execution.get("adapt_pool_label_filter_json") is not None
            or execution.get("ra_phase0_gradient_shortlist_policy")
            != PHASE0_POLICY
            or execution.get("ra_phase0_gradient_shortlist_size")
            != PHASE0_SHORTLIST_SIZE
            or execution.get("phase1_prune_enabled") is not True
            or invariants.get("beam_shape")
            != "three_live_two_children_per_parent_v1"
            or invariants.get("plateau_prior_mean_decrease_ratio_threshold")
            != 1.0e-4
            or invariants.get("physical_operator_lanes_active") is not False
            or invariants.get("shortlist_population_policy")
            != "single_global_population_v1"
            or invariants.get("phase0_active") is not True
            or invariants.get("phase0_score")
            != "standard_adapt_absolute_gradient_v1"
            or invariants.get("phase0_fubini_metric_active") is not False
            or invariants.get("phase0_resource_cost_active") is not False
            or invariants.get("phase0_compile_cost_active") is not False
            or invariants.get("phase0_shortlist_size")
            != PHASE0_SHORTLIST_SIZE
            or invariants.get("phase0_estimator_components") != ["N_grad"]
            or invariants.get("selector_qiskit_compile_cost_active") is not False
            or invariants.get("phase_i_phase_ii_phase_iii_cost_source")
            != STRUCTURAL_PROXY_COST_SOURCE
            or invariants.get("macro_generator_identity_preserved_all_phases")
            is not True
            or invariants.get("singleton_child_exposure_active") is not False
            or invariants.get("same_cutoff_reference_required") is not True
            or invariants.get("candidate_funnel_order")
            != EXPECTED_CANDIDATE_FUNNEL
            or any(
                key in invariants
                for key in (
                    "selector_compile_cost_policy",
                    "selector_compile_cost_phase_reuse",
                    "phase_ii_compile_cost_source",
                    "phase_iii_compile_cost_source",
                    "post_exposure_singleton_phase_i_policy",
                )
            )
        ):
            raise PackageContractError(f"Target route drifted: {cell.cell_id}")
        protocol_path = BUNDLE_ROOT / "protocols" / f"{cell.cell_id}.json"
        _write_json(protocol_path, protocol.to_dict())
        protocol_binding = {
            "execution_id": cell.cell_id,
            **binding(protocol_path, root=PACKAGE_DIR, canonical=True),
        }
        protocol_bindings.append(protocol_binding)
        route_digest = str(route["sha256"])
        route_digests.add(route_digest)
        artifact_contract = expected["cells"][cell.cell_id][
            "expected_run_artifacts"
        ]
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "execution_id": cell.cell_id,
                "cell_id": cell.cell_id,
                "regime_id": cell.regime_id,
                "nph": cell.nph,
                "target_horizon": cell.horizon,
                "algorithm_id": ALGORITHM_ID,
                "route_id": ROUTE_ID,
                "route_contract_sha256": route_digest,
                "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
                "structural_proxy_mode": STRUCTURAL_PROXY_MODE,
                "structural_proxy_cost_source": STRUCTURAL_PROXY_COST_SOURCE,
                "selector_qiskit_compile_cost_active": False,
                "protocol_path": protocol_binding["path"],
                "protocol_file_sha256": protocol_binding["sha256"],
                "protocol_sha256": protocol.sha256,
                "bundle_manifest_sha256": bundle_manifest["sha256"],
                "source_locks_sha256": locks["sha256"],
                "implementation_source_inventory_sha256": locks[
                    "implementation_sources"
                ]["sha256"],
                "expected_artifacts_manifest_sha256": expected["sha256"],
                "expected_run_artifacts": artifact_contract,
                "resources": RESOURCE_ENVELOPES[cell.regime_id],
                "expected_output_archive": f"{cell.cell_id}.tar.gz",
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
        job_path = PACKAGE_DIR / "jobs" / f"{cell.cell_id}.json"
        _write_json(job_path, job)
        jobs.append(job)
    if len(route_digests) != 1:
        raise PackageContractError("Six-regime route digest is not common.")

    validation = digested(
        {
            "schema": "macro_gradient_phase0_proxy_no_lanes_validation_v2",
            "status": "passed",
            "bundle_id": BUNDLE_ID,
            "protocol_count": len(cells),
            "route_contract_sha256": next(iter(route_digests)),
            "qiskit_selector_cost_phases": [],
            "phase_i_phase_ii_phase_iii_cost_source": (
                STRUCTURAL_PROXY_COST_SOURCE
            ),
            "phase0_policy": PHASE0_POLICY,
            "phase0_shortlist_size": PHASE0_SHORTLIST_SIZE,
            "phase0_estimator_components": ["N_grad"],
            "phase0_fubini_metric_active": False,
            "selector_qiskit_compile_cost_active": False,
            "macro_generator_identity_preserved_all_phases": True,
            "singleton_child_exposure_active": False,
            "lane_shortlisting_disabled": True,
            "plateau_prior_mean_decrease_ratio_threshold": 1.0e-4,
            "all_horizons": WEAK_HORIZON,
            "beam": request.method.beam.to_dict(),
            "pruning": request.method.pruning.to_dict(),
        }
    )
    _write_json(BUNDLE_ROOT / "validation_report.json", validation)
    source_manifest = _write_source_archive(locks)
    queue_path = PACKAGE_DIR / "queue.tsv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("xb") as stream:
        stream.write(
            (
                "\n".join(
                    "\t".join(
                        (
                            job["execution_id"],
                            f"jobs/{job['execution_id']}.json",
                            job["protocol_path"],
                            sha256_file(
                                PACKAGE_DIR / "jobs" / f"{job['execution_id']}.json"
                            ),
                            str(job["resources"]["request_cpus"]),
                            str(job["resources"]["request_memory_mb"]),
                            str(job["resources"]["request_disk_mb"]),
                            str(job["resources"]["max_runtime_seconds"]),
                        )
                    )
                    for job in jobs
                )
                + "\n"
            ).encode("utf-8")
        )
    plan = digested(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": len(jobs),
            "execution_ids": [job["execution_id"] for job in jobs],
            "route_contract_sha256": next(iter(route_digests)),
            "source_archive_sha256": source_manifest["archive"]["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "execution_plan.json", plan)
    audit = digested(
        {
            "schema": "macro_gradient_phase0_proxy_no_lanes_source_audit_v2",
            "status": "passed",
            "source_route_profile": SOURCE_ROUTE_PROFILE,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "target_route_profile": TARGET_ROUTE_PROFILE,
            "target_route_contract_sha256": next(iter(route_digests)),
            "expected_target_route_contract_sha256": (
                TARGET_ROUTE_CONTRACT_SHA256
            ),
            "target_parent_route_contract_sha256": (
                TARGET_PARENT_ROUTE_CONTRACT_SHA256
            ),
            "mechanical_control_plane_parent": binding(
                baseline_manifest_path,
                root=REPO_ROOT,
                canonical=True,
            ),
            "scientific_source_final_receipt": binding(
                source_receipt_path,
                root=REPO_ROOT,
                canonical=True,
            ),
            "scientific_source_locks": binding(
                SOURCE_BUNDLE_ROOT / "source_locks.json",
                root=REPO_ROOT,
                canonical=True,
            ),
            "scientific_source_protocols": source_protocol_bindings,
            "phase0_policy": PHASE0_POLICY,
            "phase0_shortlist_size": PHASE0_SHORTLIST_SIZE,
            "candidate_representation": CANDIDATE_REPRESENTATION,
            "selector_qiskit_compile_cost_active": False,
            "physical_operator_lanes_active": False,
            "structural_proxy_mode": STRUCTURAL_PROXY_MODE,
            "beam": {
                "kind": "fork_local",
                "live_parent_branches": BEAM_LIVE_BRANCHES,
                "admission_children_per_parent": BEAM_CHILDREN_PER_PARENT,
                "maximum_admission_children_per_round": (
                    BEAM_MAXIMUM_CHILDREN_PER_ROUND
                ),
                "s_alg_weight": BEAM_S_ALG_WEIGHT,
            },
            "pruning": PRUNING_POLICY,
            "implementation_source_inventory_sha256": locks[
                "implementation_sources"
            ]["sha256"],
            "scientific_result_anchor_claimed": False,
        }
    )
    _write_json(PACKAGE_DIR / "source_lock_audit.json", audit)
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "passed_inert_six_cells",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "row_count": len(jobs),
            "execution_ids": [job["execution_id"] for job in jobs],
            "child_route_contract_sha256": next(iter(route_digests)),
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "implementation_source_inventory_sha256": locks[
                "implementation_sources"
            ]["sha256"],
            "bundle_manifest": binding(
                BUNDLE_ROOT / "bundle_manifest.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_expected_artifacts": binding(
                BUNDLE_ROOT / "expected_artifacts.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_source_locks": binding(
                BUNDLE_ROOT / "source_locks.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_validation_report": binding(
                BUNDLE_ROOT / "validation_report.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "protocols": protocol_bindings,
            "jobs": [
                {
                    "execution_id": job["execution_id"],
                    **binding(
                        PACKAGE_DIR / "jobs" / f"{job['execution_id']}.json",
                        root=PACKAGE_DIR,
                        canonical=True,
                    ),
                }
                for job in jobs
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
                PACKAGE_DIR / "execution_plan.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "source_lock_audit": binding(
                PACKAGE_DIR / "source_lock_audit.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "control_files": [
                binding(PACKAGE_DIR / name, root=PACKAGE_DIR)
                for name in CONTROL_FILES
            ],
            "required_route_source_paths": list(
                REQUIRED_ROUTE_SOURCE_PATHS
            ),
            "remote_image_path": REMOTE_IMAGE_PATH,
            "remote_image_sha256": REMOTE_IMAGE_SHA256,
            "weak_holstein_horizon": WEAK_HORIZON,
            "strong_holstein_horizon": STRONG_HORIZON,
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
        "status": "passed_inert_six_cells",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "route_contract_sha256": next(iter(route_digests)),
        "source_archive_sha256": source_manifest["archive"]["sha256"],
        "row_count": 6,
    }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
