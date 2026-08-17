#!/usr/bin/env python3
"""Seal the four-cell pure-Hubbard Page-12 low/high-noise prefix."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import subprocess
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
    ALGORITHM_SEED,
    APPLICATION_SOURCE_LOCK_KEY,
    BACKEND_COMPILE_SCOPE,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    CELL_COUNT,
    CELL_ROWS,
    COHERENT_NOISE_SEED,
    CONTROL_FILES,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    EXPECTED_CANDIDATE_FUNNEL,
    GENERATED_PATHS,
    INSERTION_POLICY,
    INERT_PACKAGE_STATUS,
    JOB_SCHEMA,
    NOISE_TUPLE_ORDER,
    NOISE_LEVELS,
    OPTIMIZER,
    OPTIMIZER_MAXITER,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    P3_RECEIPT_SCHEMA,
    P4_RECEIPT_SCHEMA,
    PHASE0_POLICY,
    PHASE0_SHORTLIST_SIZE,
    PLATEAU_THRESHOLD,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REQUIRED_ROUTE_SOURCE_PATHS,
    RESOURCE_ENVELOPE,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_ID,
    RUN_CLASS,
    SELECTOR_COMPILE_COST_PHASE_REUSE,
    SELECTOR_COMPILE_COST_POLICY,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    STAGE_ID,
    SOURCE_HORIZON,
    SOURCE_IMPLEMENTATION_INVENTORY_SHA256,
    SOURCE_PACKAGE_ID,
    SOURCE_PACKAGE_MANIFEST_SHA256,
    SOURCE_PACKAGE_RELATIVE_PATH,
    SOURCE_REQUEST_MEMORY_MB,
    TARGET_HORIZON,
    TARGET_ROUTE_PROFILE,
    VALUE_NOISE_SEED,
    PackageContractError,
    binding,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    execution_id,
    load_json,
    repo_root_from_script,
    reject_cache_artifacts,
    sha256_file,
    source_lock_id,
    verify_self_digest,
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
from pipelines.static_adapt.ra_adapt.pure_hubbard_noise_page12 import (  # noqa: E402
    PAPER_I_PURE_HUBBARD_NOISE_COHERENT_SEED,
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID,
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID,
    PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY,
    PAPER_I_PURE_HUBBARD_NOISE_VALUE_SEED,
    build_paper_i_pure_hubbard_noise_page12_problem,
    build_paper_i_pure_hubbard_noise_page12_request,
    paper_i_pure_hubbard_noise_page12_application_source_contract,
    pure_hubbard_noise_level_contract,
)


BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _cells() -> tuple[BundleCellSpec, ...]:
    return tuple(
        BundleCellSpec(
            cell_id=execution_id(u_value, noise_level),
            stage=STAGE_ID,
            regime_id=f"pure_hubbard_u{str(u_value).replace('.', 'p')}",
            nph=0,
            route_id=ROUTE_ID,
            algorithm_id=ALGORITHM_ID,
            selector_family="ra_adapt",
            candidate_representation=CANDIDATE_REPRESENTATION,
            horizon=TARGET_HORIZON,
            source_lock_id=source_lock_id(u_value, noise_level),
        )
        for u_value, noise_level, _ in CELL_ROWS
    )


def _application_sources() -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for u_value, noise_level, expected_tuple in CELL_ROWS:
        cell_id = execution_id(u_value, noise_level)
        problem = build_paper_i_pure_hubbard_noise_page12_problem(u=u_value)
        request = build_paper_i_pure_hubbard_noise_page12_request(
            noise_level=noise_level,
            maximum_controller_rounds=TARGET_HORIZON,
        )
        source = paper_i_pure_hubbard_noise_page12_application_source_contract(
            problem,
            request,
        )
        if (
            source.get("algorithm_id") != ALGORITHM_ID
            or source.get("adapter_id") != CANDIDATE_ADAPTER_ID
            or tuple(source.get("noise", {}).get("noise_tuple", ()))
            != tuple(expected_tuple)
            or tuple(source.get("noise", {}).get("noise_tuple_order", ()))
            != NOISE_TUPLE_ORDER
            or source.get("scientific_settings", {}).get("maximum_controller_rounds")
            != TARGET_HORIZON
            or source.get("scientific_settings", {}).get("insertion")
            != INSERTION_POLICY
        ):
            raise PackageContractError(
                f"Application source contract drifted for {cell_id}."
            )
        sources[cell_id] = source
    if (
        len(sources) != CELL_COUNT
        or len({row["sha256"] for row in sources.values()}) != CELL_COUNT
    ):
        raise PackageContractError("The application sources are not unique.")
    return sources


def _source_locks(
    *,
    cells: tuple[BundleCellSpec, ...],
    implementation: Mapping[str, Any],
    application_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cell_locks: dict[str, dict[str, Any]] = {}
    application_bindings: dict[str, dict[str, Any]] = {}
    row_by_id = {
        execution_id(u_value, noise_level): (u_value, noise_level, noise_tuple)
        for u_value, noise_level, noise_tuple in CELL_ROWS
    }
    for cell in cells:
        u_value, noise_level, noise_tuple = row_by_id[cell.cell_id]
        source = application_sources[cell.cell_id]
        source_path = (
            PACKAGE_DIR
            / "source_authority"
            / f"{cell.cell_id}.application_source_contract.json"
        )
        _write_json(source_path, source)
        application_bindings[cell.cell_id] = binding(
            source_path,
            root=PACKAGE_DIR,
            canonical=True,
        )
        exact_reference = dict(source["same_cutoff_exact_reference"])
        cell_locks[cell.source_lock_id] = digested(
            {
                "schema": "paper_i_pure_hubbard_page12_noise_cell_source_lock_v1",
                "source_lock_id": cell.source_lock_id,
                "cell_id": cell.cell_id,
                "u_over_t": float(u_value),
                "noise_level_id": noise_level,
                "noise_tuple_order": list(NOISE_TUPLE_ORDER),
                "noise_tuple": list(noise_tuple),
                "application_source_contract_sha256": source["sha256"],
                "problem_request_sha256": source["problem_request_sha256"],
                "same_cutoff_exact_reference_sha256": canonical_sha256(
                    exact_reference
                ),
            }
        )
    return digested(
        {
            "schema": "paper_i_pure_hubbard_page12_noise_source_locks_v1",
            "package_id": PACKAGE_ID,
            "implementation_sources": dict(implementation),
            "application_source_contracts": application_bindings,
            "cell_locks": cell_locks,
            "fixed_non_noise_settings": {
                "maximum_controller_rounds": TARGET_HORIZON,
                "optimizer": OPTIMIZER,
                "optimizer_maxiter": OPTIMIZER_MAXITER,
                "algorithm_seed": ALGORITHM_SEED,
                "value_noise_seed": VALUE_NOISE_SEED,
                "coherent_noise_seed": COHERENT_NOISE_SEED,
                "plateau_threshold": PLATEAU_THRESHOLD,
                "insertion_policy": INSERTION_POLICY,
            },
        }
    )


def _source_lock_refs(
    *,
    cell: BundleCellSpec,
    locks: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, str]:
    cell_lock = locks["cell_locks"][cell.source_lock_id]
    return {
        "source_locks_manifest_sha256": str(locks["sha256"]),
        "implementation_source_inventory_sha256": str(
            locks["implementation_sources"]["sha256"]
        ),
        "cell_source_lock_id": cell.source_lock_id,
        "cell_source_lock_sha256": str(cell_lock["sha256"]),
        "ed_cutoff_reference_sha256": str(
            cell_lock["same_cutoff_exact_reference_sha256"]
        ),
        APPLICATION_SOURCE_LOCK_KEY: str(source["sha256"]),
    }


def _write_source_archive(source_locks: Mapping[str, Any]) -> dict[str, Any]:
    implementation = source_locks["implementation_sources"]
    rows = [
        {
            "path": str(raw["path"]),
            "sha256": str(raw["sha256"]),
            "size_bytes": int((REPO_ROOT / str(raw["path"])).stat().st_size),
            "source_kind": "verified_current_implementation_inventory",
        }
        for raw in implementation["files"]
    ]
    rows.sort(key=lambda row: row["path"])
    observed = {row["path"] for row in rows}
    missing = sorted(set(REQUIRED_ROUTE_SOURCE_PATHS) - observed)
    if missing:
        raise PackageContractError(
            "Implementation inventory omitted required noise-route source: "
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


def _run_preflight(mode: str, *, job_path: Path | None = None) -> dict[str, Any]:
    output = PACKAGE_DIR / (
        "p3_numerical_receipt.json"
        if mode == "p3"
        else "p4_packaged_numerical_receipt.json"
    )
    command = [
        sys.executable,
        "-B",
        (PACKAGE_DIR / "run_numerical_preflight.py").as_posix(),
        "--mode",
        mode,
        "--output",
        output.as_posix(),
    ]
    if job_path is not None:
        command.extend(("--job", job_path.as_posix()))
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
    environment["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            f"{mode.upper()} numerical preflight failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    receipt = load_json(output, label=f"{mode.upper()} numerical receipt")
    verify_self_digest(receipt, label=f"{mode.upper()} numerical receipt")
    expected_schema = P3_RECEIPT_SCHEMA if mode == "p3" else P4_RECEIPT_SCHEMA
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("status") != "passed"
        or receipt.get("scientific_execution_performed") is not True
        or receipt.get("real_noisy_gradient_probe_passed") is not True
        or receipt.get("real_noisy_powell_probe_passed") is not True
        or int(receipt.get("completed_controller_rounds", -1)) != 1
    ):
        raise PackageContractError(f"{mode.upper()} numerical receipt drifted.")
    return receipt


def build() -> dict[str, Any]:
    reject_cache_artifacts(PACKAGE_DIR)
    if any((PACKAGE_DIR / name).exists() for name in GENERATED_PATHS):
        raise FileExistsError("Refusing to overwrite an existing package seal.")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file():
            raise PackageContractError(f"Missing control file: {name}")
    if (
        ALGORITHM_ID != PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ALGORITHM_ID
        or CANDIDATE_ADAPTER_ID
        != PAPER_I_PURE_HUBBARD_NOISE_PAGE12_ADAPTER_ID
        or APPLICATION_SOURCE_LOCK_KEY
        != PAPER_I_PURE_HUBBARD_NOISE_PAGE12_SOURCE_LOCK_KEY
        or VALUE_NOISE_SEED != PAPER_I_PURE_HUBBARD_NOISE_VALUE_SEED
        or COHERENT_NOISE_SEED != PAPER_I_PURE_HUBBARD_NOISE_COHERENT_SEED
    ):
        raise PackageContractError("Package constants drifted from the named seam.")
    for _u, level, expected_tuple in CELL_ROWS:
        if tuple(pure_hubbard_noise_level_contract(level)["noise_tuple"]) != tuple(
            expected_tuple
        ):
            raise PackageContractError(f"Noise tuple drifted for {level}.")

    p3 = _run_preflight("p3")
    cells = _cells()
    application_sources = _application_sources()
    implementation = _implementation_source_inventory(REPO_ROOT)
    source_package_manifest = load_json(
        REPO_ROOT / SOURCE_PACKAGE_RELATIVE_PATH / "package_manifest.json",
        label="source package manifest",
    )
    verify_self_digest(source_package_manifest, label="source package manifest")
    if (
        source_package_manifest.get("package_id") != SOURCE_PACKAGE_ID
        or source_package_manifest.get("sha256")
        != SOURCE_PACKAGE_MANIFEST_SHA256
        or source_package_manifest.get("implementation_source_inventory_sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
        or implementation["sha256"]
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
    ):
        raise PackageContractError(
            "The source package or implementation inventory drifted."
        )
    if (
        p3.get("implementation_source_inventory_sha256")
        != implementation["sha256"]
        or p3.get("application_source_contract_sha256s")
        != {
            cell_id: row["sha256"]
            for cell_id, row in sorted(application_sources.items())
        }
    ):
        raise PackageContractError("P3 did not bind the seal-time source state.")
    locks = _source_locks(
        cells=cells,
        implementation=implementation,
        application_sources=application_sources,
    )
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=False)
    _write_json(BUNDLE_ROOT / "source_locks.json", locks)
    expected = digested(
        {
            "schema": "paper_i_pure_hubbard_page12_noise_expected_artifacts_v1",
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
            "schema": "paper_i_pure_hubbard_page12_noise_bundle_manifest_v1",
            "bundle_id": BUNDLE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "cell_count": CELL_COUNT,
            "cells": [cell.to_dict() for cell in cells],
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

    row_by_id = {
        execution_id(u_value, noise_level): (u_value, noise_level, noise_tuple)
        for u_value, noise_level, noise_tuple in CELL_ROWS
    }
    protocol_bindings: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    route_digests_by_level: dict[str, set[str]] = {}
    for cell in cells:
        u_value, noise_level, expected_tuple = row_by_id[cell.cell_id]
        problem = build_paper_i_pure_hubbard_noise_page12_problem(u=u_value)
        request = build_paper_i_pure_hubbard_noise_page12_request(
            noise_level=noise_level,
            maximum_controller_rounds=TARGET_HORIZON,
        )
        source = application_sources[cell.cell_id]
        refs = _source_lock_refs(
            cell=cell,
            locks=locks,
            source=source,
        )
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
        execution = route.get("execution_settings") if isinstance(route, Mapping) else None
        invariants = route.get("semantic_invariants") if isinstance(route, Mapping) else None
        noise = execution.get("ra_controller_noise_contract") if isinstance(execution, Mapping) else None
        if (
            protocol.algorithm_id != ALGORITHM_ID
            or protocol.adapter_id != CANDIDATE_ADAPTER_ID
            or protocol.horizon != TARGET_HORIZON
            or protocol.problem.problem_key != "hubbard"
            or protocol.problem.num_sites != 2
            or protocol.problem.n_ph_max != 0
            or protocol.problem.n_fermions != 2
            or protocol.problem.u != float(u_value)
            or protocol.source_locks.get(APPLICATION_SOURCE_LOCK_KEY)
            != source["sha256"]
            or not isinstance(route, Mapping)
            or route.get("route_profile") != TARGET_ROUTE_PROFILE
            or not isinstance(execution, Mapping)
            or not isinstance(invariants, Mapping)
            or not isinstance(noise, Mapping)
            or tuple(noise.get("noise_tuple", ())) != tuple(expected_tuple)
            or noise.get("noise_level_id") != noise_level
            or execution.get("adapt_parallel_gradient_workers") != 1
            or execution.get("phase3_backend_cost_scope") != BACKEND_COMPILE_SCOPE
            or execution.get("static_lane_route") != "global_single_population"
            or execution.get("ra_phase0_gradient_shortlist_policy")
            != PHASE0_POLICY
            or execution.get("ra_phase0_gradient_shortlist_size")
            != PHASE0_SHORTLIST_SIZE
            or invariants.get("application_lane")
            != "paper_i_pure_hubbard_page12_full_noise_v1"
            or invariants.get("controller_noise_active") is not True
            or invariants.get("controller_noise_candidate_gradient_scoring")
            != "noisy"
            or invariants.get("controller_noise_powell_refit_objective")
            != "noisy"
            or invariants.get("controller_noise_geometry_and_gram") != "exact"
            or invariants.get("candidate_funnel_order")
            != EXPECTED_CANDIDATE_FUNNEL
            or invariants.get("physical_operator_lanes_active") is not False
            or invariants.get("phase0_active") is not True
            or invariants.get("phase0_score")
            != "standard_adapt_absolute_gradient_v1"
            or invariants.get("phase0_shortlist_size") != PHASE0_SHORTLIST_SIZE
            or invariants.get("phase_i_compile_cost_source")
            != "structural_proxy_v1"
            or invariants.get("phase_ii_compile_cost_source")
            != "backend_transpile_v1"
            or invariants.get("phase_iii_compile_cost_source")
            != "backend_transpile_v1"
            or invariants.get("plateau_cumulative_decrease_ratio_threshold")
            != PLATEAU_THRESHOLD
            or "plateau_prior_mean_decrease_ratio_threshold" in invariants
            or invariants.get("plateau_threshold_comparison")
            != "marginal_to_prior_cumulative_strictly_below_v1"
            or invariants.get("plateau_trigger_source")
            != (
                "immediately_preceding_marginal_over_prior_cumulative_"
                "accepted_post_full_refit_energy_decrease_v1"
            )
            or invariants.get("plateau_threshold_calibration_status")
            != "source_locked_completed_trajectory_replay_v1"
            or invariants.get("plateau_energy_source")
            != "persisted_noisy_controller_energy_before_after_v1"
            or invariants.get("selector_compile_cost_policy")
            != SELECTOR_COMPILE_COST_POLICY
            or invariants.get("selector_compile_cost_phase_reuse")
            != SELECTOR_COMPILE_COST_PHASE_REUSE
        ):
            raise PackageContractError(f"Named protocol drifted: {cell.cell_id}")
        route_digest = str(route["sha256"])
        route_digests_by_level.setdefault(noise_level, set()).add(route_digest)
        protocol_path = BUNDLE_ROOT / "protocols" / f"{cell.cell_id}.json"
        _write_json(protocol_path, protocol.to_dict())
        protocol_binding = {
            "execution_id": cell.cell_id,
            **binding(protocol_path, root=PACKAGE_DIR, canonical=True),
        }
        protocol_bindings.append(protocol_binding)
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "execution_id": cell.cell_id,
                "cell_id": cell.cell_id,
                "u_over_t": float(u_value),
                "noise_level_id": noise_level,
                "noise_tuple_order": list(NOISE_TUPLE_ORDER),
                "noise_tuple": list(expected_tuple),
                "num_sites": 2,
                "nph": 0,
                "target_horizon": TARGET_HORIZON,
                "algorithm_id": ALGORITHM_ID,
                "route_id": ROUTE_ID,
                "source_lock_id": cell.source_lock_id,
                "route_contract_sha256": route_digest,
                "application_source_contract_sha256": source["sha256"],
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
                "expected_run_artifacts": _expected_artifacts(cell.cell_id),
                "resources": dict(RESOURCE_ENVELOPE),
                "output_archive_contract": {
                    "schema": (
                        "paper_i_pure_hubbard_page12_noise_"
                        "terminal_archive_v1"
                    ),
                    "path_template": (
                        "transfer/{execution_id}__{cluster_id}__"
                        "{proc_id}.tar.gz"
                    ),
                    "transfer_policy": "on_exit_only_v1",
                    "scheduler_attempt_identity_inside_archive": True,
                },
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
    expected_noise_levels = {level for _u, level, _tuple in CELL_ROWS}
    if set(route_digests_by_level) != expected_noise_levels or any(
        len(values) != 1 for values in route_digests_by_level.values()
    ):
        raise PackageContractError("Route digest must be common across U per noise rung.")

    validation = digested(
        {
            "schema": "paper_i_pure_hubbard_page12_noise_validation_v1",
            "status": "passed",
            "bundle_id": BUNDLE_ID,
            "protocol_count": CELL_COUNT,
            "route_contract_sha256_by_noise_level": {
                level: next(iter(values))
                for level, values in sorted(route_digests_by_level.items())
            },
            "u_over_t_values": [1.5, 8.0],
            "noise_levels": [level for level, _tuple in NOISE_LEVELS],
            "qiskit_cost_phases": ["phase_ii", "phase_iii"],
            "lane_shortlisting_disabled": True,
            "plateau_energy_source": (
                "persisted_noisy_controller_energy_before_after_v1"
            ),
            "plateau_cumulative_decrease_ratio_threshold": PLATEAU_THRESHOLD,
            "target_horizon": TARGET_HORIZON,
            "p3_receipt_sha256": p3["sha256"],
        }
    )
    _write_json(BUNDLE_ROOT / "validation_report.json", validation)
    source_manifest = _write_source_archive(locks)
    queue_path = PACKAGE_DIR / "queue.tsv"
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
            "row_count": CELL_COUNT,
            "execution_ids": [job["execution_id"] for job in jobs],
            "route_contract_sha256_by_noise_level": {
                level: next(iter(values))
                for level, values in sorted(route_digests_by_level.items())
            },
            "application_source_contract_sha256s": {
                cell_id: row["sha256"]
                for cell_id, row in sorted(application_sources.items())
            },
            "source_archive_sha256": source_manifest["archive"]["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "execution_plan.json", plan)
    p4_job = PACKAGE_DIR / "jobs" / f"{jobs[0]['execution_id']}.json"
    p4 = _run_preflight("p4", job_path=p4_job)
    if (
        p4.get("source_archive_sha256") != source_manifest["archive"]["sha256"]
        or p4.get("job_spec_sha256") != jobs[0]["sha256"]
        or p4.get("protocol_sha256") != jobs[0]["protocol_sha256"]
    ):
        raise PackageContractError("P4 did not bind the packaged execution seam.")

    audit = digested(
        {
            "schema": "paper_i_pure_hubbard_page12_noise_source_lock_audit_v1",
            "status": "passed",
            "study_shape": "u2_by_low_high_noise2_r30_recovery_v1",
            "source_package": {
                "package_id": SOURCE_PACKAGE_ID,
                "path": SOURCE_PACKAGE_RELATIVE_PATH,
                "package_manifest_sha256": SOURCE_PACKAGE_MANIFEST_SHA256,
                "implementation_source_inventory_sha256": (
                    SOURCE_IMPLEMENTATION_INVENTORY_SHA256
                ),
                "target_horizon": SOURCE_HORIZON,
                "request_memory_mb": SOURCE_REQUEST_MEMORY_MB,
            },
            "fixed_fields": [
                "physics_except_u_over_t",
                "page12_route",
                "optimizer_and_budget",
                "algorithm_and_noise_seeds",
                "plateau_policy_and_threshold",
            ],
            "scientific_changed_fields_vs_source": [
                "maximum_controller_rounds"
            ],
            "operational_changed_fields_vs_source": [
                "planned_execution_subset",
                "request_memory_mb",
            ],
            "source_target_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "source_request_memory_mb": SOURCE_REQUEST_MEMORY_MB,
            "request_memory_mb": RESOURCE_ENVELOPE["request_memory_mb"],
            "omitted_completed_noise_level_ids": ["extreme"],
            "matrix_fields": ["u_over_t", "noise_level_id"],
            "application_source_contract_sha256s": {
                cell_id: row["sha256"]
                for cell_id, row in sorted(application_sources.items())
            },
            "implementation_source_inventory_sha256": implementation["sha256"],
            "source_implementation_inventory_match": (
                implementation["sha256"]
                == SOURCE_IMPLEMENTATION_INVENTORY_SHA256
            ),
            "p3_receipt_sha256": p3["sha256"],
            "p4_receipt_sha256": p4["sha256"],
            "scientific_result_anchor_claimed": False,
        }
    )
    _write_json(PACKAGE_DIR / "source_lock_audit.json", audit)
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": INERT_PACKAGE_STATUS,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "row_count": CELL_COUNT,
            "execution_ids": [job["execution_id"] for job in jobs],
            "route_contract_sha256_by_noise_level": {
                level: next(iter(values))
                for level, values in sorted(route_digests_by_level.items())
            },
            "application_source_contract_sha256s": {
                cell_id: row["sha256"]
                for cell_id, row in sorted(application_sources.items())
            },
            "implementation_source_inventory_sha256": implementation["sha256"],
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
            "application_source_contracts": [
                {
                    "execution_id": cell_id,
                    **binding(
                        PACKAGE_DIR
                        / "source_authority"
                        / f"{cell_id}.application_source_contract.json",
                        root=PACKAGE_DIR,
                        canonical=True,
                    ),
                }
                for cell_id in sorted(application_sources)
            ],
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
                PACKAGE_DIR / "execution_plan.json", root=PACKAGE_DIR, canonical=True
            ),
            "source_lock_audit": binding(
                PACKAGE_DIR / "source_lock_audit.json", root=PACKAGE_DIR, canonical=True
            ),
            "p3_numerical_receipt": binding(
                PACKAGE_DIR / "p3_numerical_receipt.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "p4_packaged_numerical_receipt": binding(
                PACKAGE_DIR / "p4_packaged_numerical_receipt.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "control_files": [
                binding(PACKAGE_DIR / name, root=PACKAGE_DIR)
                for name in CONTROL_FILES
            ],
            "required_route_source_paths": list(REQUIRED_ROUTE_SOURCE_PATHS),
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
    reject_cache_artifacts(PACKAGE_DIR)
    return {
        "status": INERT_PACKAGE_STATUS,
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "source_archive_sha256": source_manifest["archive"]["sha256"],
        "p3_receipt_sha256": p3["sha256"],
        "p4_receipt_sha256": p4["sha256"],
        "row_count": CELL_COUNT,
    }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
