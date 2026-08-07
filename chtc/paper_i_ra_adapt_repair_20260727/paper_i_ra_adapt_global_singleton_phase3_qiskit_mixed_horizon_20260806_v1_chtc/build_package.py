#!/usr/bin/env python3
"""Build the inert six-cell Phase-III-only Qiskit CHTC package."""

from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Sequence


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
    BACKEND_COMPILE_SCOPE,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    CONTROL_FILES,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    GENERATED_PATHS,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PREDECESSOR_SOURCE_LOCKS,
    PREDECESSOR_SOURCE_LOCKS_CANONICAL_SHA256,
    PREDECESSOR_SOURCE_LOCKS_FILE_SHA256,
    PROBLEM_BASELINES,
    PROBLEM_BASELINES_FILE_SHA256,
    REGIME_ROWS,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REQUIRED_PHASE3_QISKIT_SOURCE_PATHS,
    RESOURCE_ENVELOPES,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_ID,
    RUN_CLASS,
    SELECTOR_COMPILE_COST_PHASE_REUSE,
    SELECTOR_COMPILE_COST_POLICY,
    SOURCE_ALGORITHM_ID,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    SOURCE_AUTHORITY_SCHEMA,
    SOURCE_LOCK_AUDIT_SCHEMA,
    SOURCE_PROTOCOLS,
    SOURCE_ROUTE_CONTRACT_SHA256,
    SOURCE_ROUTE_PROFILE,
    STRONG_SOURCE_PACKAGE,
    STRONG_SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    STRONG_SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    TARGET_ROUTE_PROFILE,
    WEAK_SOURCE_PACKAGE,
    WEAK_SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    WEAK_SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    PackageContractError,
    binding,
    canonical_json_bytes,
    digested,
    execution_id,
    load_json,
    repo_root_from_script,
    sha256_file,
    source_lock_id,
    verify_self_digest,
)


REPO_ROOT = repo_root_from_script(__file__)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    materialize_stationary_core_v12 as v12,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    PHASE3_QISKIT_BUNDLE_ID,
    PHASE3_QISKIT_CAMPAIGN_ID,
    PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256,
    PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE,
    _implementation_source_inventory,
    build_phase3_qiskit_mixed_horizon_cell_specs,
    load_validated_bundle_protocol,
    materialize_phase3_qiskit_mixed_horizon_bundle,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    canonical_sha256,
)


support = v12.support
AUTHORITY_ARCHIVE_RELATIVE = Path(
    "source_authority/page7_parent_protocols.tar.gz"
)
AUTHORITY_MANIFEST_RELATIVE = Path(
    "source_authority/page7_parent_protocol_authority.json"
)
PREDECESSOR_LOCKS_COPY_RELATIVE = Path(
    "source_authority/predecessor_source_locks.json"
)
PROBLEM_BASELINES_COPY_RELATIVE = Path(
    "source_authority/problem_baselines.json"
)
BUNDLE_ROOT = PACKAGE_DIR / "bundle_materialization" / BUNDLE_ID


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise PackageContractError(f"Missing or unsafe source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256_file(source) != sha256_file(destination):
        raise PackageContractError(f"Exact copy drifted: {destination}")


def _load_bound_digested(
    path: Path,
    *,
    file_sha256: str,
    canonical_sha256_value: str,
    label: str,
) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != file_sha256
    ):
        raise PackageContractError(f"{label} exact bytes drifted.")
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != canonical_sha256_value:
        raise PackageContractError(f"{label} canonical digest drifted.")
    return payload


def _repository_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "git_commit": commit,
        "dirty_working_tree": dirty,
        "cwd": REPO_ROOT.as_posix(),
    }


def _write_deterministic_archive(
    *,
    destination: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for row in rows:
                    source = Path(str(row["source_path"]))
                    arcname = str(row["path"])
                    if (
                        not source.is_file()
                        or source.is_symlink()
                        or source.stat().st_size != int(row["size_bytes"])
                        or sha256_file(source) != row["sha256"]
                    ):
                        raise PackageContractError(
                            f"Archive source drifted: {source}"
                        )
                    info = archive.gettarinfo(str(source), arcname=arcname)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with source.open("rb") as stream:
                        archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())


def _verify_source_packages() -> dict[str, Any]:
    weak = _load_bound_digested(
        REPO_ROOT / WEAK_SOURCE_PACKAGE / "package_manifest.json",
        file_sha256=WEAK_SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
        canonical_sha256_value=(
            WEAK_SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        label="sealed page-7 weak-sector package manifest",
    )
    strong = _load_bound_digested(
        REPO_ROOT / STRONG_SOURCE_PACKAGE / "package_manifest.json",
        file_sha256=STRONG_SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
        canonical_sha256_value=(
            STRONG_SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        label="sealed page-7 strong-sector r70 package manifest",
    )
    if (
        weak.get("execution_authorized") is not False
        or weak.get("submitted") is not False
        or strong.get("execution_authorized") is not False
        or strong.get("submitted") is not False
    ):
        raise PackageContractError(
            "A page-7 source package lost its inert manifest state."
        )
    return {"weak": weak, "strong": strong}


def _source_protocol_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_regimes = [regime for regime, _nph, _horizon in REGIME_ROWS]
    if [str(row["regime_id"]) for row in SOURCE_PROTOCOLS] != expected_regimes:
        raise PackageContractError("Source protocol order drifted.")
    for raw in SOURCE_PROTOCOLS:
        source_path = REPO_ROOT / Path(raw["path"])
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or sha256_file(source_path) != raw["file_sha256"]
        ):
            raise PackageContractError(
                f"Source protocol exact bytes drifted: {source_path}"
            )
        payload = load_json(source_path, label="page-7 source protocol")
        if verify_self_digest(payload, label="page-7 source protocol") != raw[
            "canonical_sha256"
        ]:
            raise PackageContractError(
                f"Source protocol canonical digest drifted: {source_path}"
            )
        route = payload.get("route_contract")
        request = payload.get("request")
        execution = (
            request.get("execution") if isinstance(request, Mapping) else None
        )
        resume = (
            execution.get("resume")
            if isinstance(execution, Mapping)
            else None
        )
        stop = (
            execution.get("stop")
            if isinstance(execution, Mapping)
            else None
        )
        problem = payload.get("problem")
        if (
            payload.get("algorithm_id") != SOURCE_ALGORITHM_ID
            or int(payload.get("horizon", -1)) != int(raw["horizon"])
            or not isinstance(route, Mapping)
            or route.get("route_profile") != SOURCE_ROUTE_PROFILE
            or route.get("sha256") != SOURCE_ROUTE_CONTRACT_SHA256
            or not isinstance(problem, Mapping)
            or int(problem.get("n_ph_max", -1)) != int(raw["nph"])
            or not isinstance(resume, Mapping)
            or resume.get("kind") != "fresh_start"
            or not isinstance(stop, Mapping)
            or int(stop.get("maximum_controller_rounds", -1))
            != int(raw["horizon"])
            or payload.get("execution_authorized") is not False
        ):
            raise PackageContractError(
                f"Page-7 source protocol semantics drifted: {source_path}"
            )
        member_path = (
            "parent_protocols/"
            f"{raw['regime_id']}__nph{raw['nph']}__r{raw['horizon']}.json"
        )
        rows.append(
            {
                **dict(raw),
                "source_path": source_path,
                "path": member_path,
                "sha256": str(raw["file_sha256"]),
                "size_bytes": source_path.stat().st_size,
                "payload": payload,
            }
        )
    return rows


def _build_source_authority(
    source_packages: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _source_protocol_rows()
    archive_path = PACKAGE_DIR / AUTHORITY_ARCHIVE_RELATIVE
    _write_deterministic_archive(destination=archive_path, rows=rows)
    archive_binding = binding(archive_path, root=PACKAGE_DIR)
    manifest = digested(
        {
            "schema": SOURCE_AUTHORITY_SCHEMA,
            "status": "passed",
            "scientific_result_anchor_claimed": False,
            "authority_role": (
                "exact_page7_resolved_protocol_route_and_settings_parent_v1"
            ),
            "source_route_profile": SOURCE_ROUTE_PROFILE,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "source_algorithm_id": SOURCE_ALGORITHM_ID,
            "weak_sector_horizon": 50,
            "strong_sector_horizon": 70,
            "source_packages": {
                "weak": {
                    "path": WEAK_SOURCE_PACKAGE.as_posix(),
                    "manifest_file_sha256": (
                        WEAK_SOURCE_PACKAGE_MANIFEST_FILE_SHA256
                    ),
                    "manifest_canonical_sha256": source_packages["weak"][
                        "sha256"
                    ],
                },
                "strong": {
                    "path": STRONG_SOURCE_PACKAGE.as_posix(),
                    "manifest_file_sha256": (
                        STRONG_SOURCE_PACKAGE_MANIFEST_FILE_SHA256
                    ),
                    "manifest_canonical_sha256": source_packages["strong"][
                        "sha256"
                    ],
                },
            },
            "archive": archive_binding,
            "protocols": [
                {
                    "regime_id": row["regime_id"],
                    "nph": row["nph"],
                    "horizon": row["horizon"],
                    "source_path": Path(row["source_path"])
                    .relative_to(REPO_ROOT)
                    .as_posix(),
                    "member_path": row["path"],
                    "file_sha256": row["sha256"],
                    "canonical_sha256": row["canonical_sha256"],
                    "route_contract_sha256": (
                        SOURCE_ROUTE_CONTRACT_SHA256
                    ),
                }
                for row in rows
            ],
            "protocol_count": len(rows),
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / AUTHORITY_MANIFEST_RELATIVE, manifest)
    return rows, manifest


def _settings_from_parent_protocol(
    payload: Mapping[str, Any],
    *,
    cell_id: str,
) -> dict[str, Any]:
    route = payload.get("route_contract")
    execution = (
        route.get("execution_settings")
        if isinstance(route, Mapping)
        else None
    )
    seeds = payload.get("seeds")
    if not isinstance(execution, Mapping) or not isinstance(seeds, Mapping):
        raise PackageContractError(
            f"Page-7 parent settings are absent for {cell_id}."
        )
    names = (
        "adapt_final_full_refit",
        "adapt_final_refit_maxiter",
        "adapt_finite_angle",
        "adapt_full_refit_every",
        "adapt_inner_optimizer",
        "adapt_maxiter",
        "adapt_reopt_policy",
        "adapt_seed",
        "adapt_window_size",
        "adapt_window_topk",
        "phase1_prune_enabled",
        "phase2_enable_batching",
        "phase3_backend_transpile_seed",
    )
    missing = [name for name in names if name not in execution]
    if missing:
        raise PackageContractError(
            f"Page-7 parent settings are incomplete for {cell_id}: {missing}."
        )
    settings = {name: copy.deepcopy(execution[name]) for name in names}
    if (
        str(payload.get("optimizer", "")).strip().lower()
        != str(settings["adapt_inner_optimizer"]).strip().lower()
        or int(payload.get("optimizer_maxiter", -1))
        != int(settings["adapt_final_refit_maxiter"])
        or int(settings["adapt_maxiter"])
        != int(settings["adapt_final_refit_maxiter"])
        or int(seeds.get("adapt", -1)) != int(settings["adapt_seed"])
        or int(seeds.get("transpiler", -1))
        != int(settings["phase3_backend_transpile_seed"])
    ):
        raise PackageContractError(
            f"Page-7 parent settings disagree internally for {cell_id}."
        )
    return settings


def _derive_source_locks(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    authority_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessor_path = REPO_ROOT / PREDECESSOR_SOURCE_LOCKS
    predecessor = _load_bound_digested(
        predecessor_path,
        file_sha256=PREDECESSOR_SOURCE_LOCKS_FILE_SHA256,
        canonical_sha256_value=(
            PREDECESSOR_SOURCE_LOCKS_CANONICAL_SHA256
        ),
        label="sealed page-7 predecessor source locks",
    )
    _copy_exact(
        predecessor_path,
        PACKAGE_DIR / PREDECESSOR_LOCKS_COPY_RELATIVE,
    )
    if (
        not PROBLEM_BASELINES.is_absolute()
        and sha256_file(REPO_ROOT / PROBLEM_BASELINES)
        == PROBLEM_BASELINES_FILE_SHA256
    ):
        _copy_exact(
            REPO_ROOT / PROBLEM_BASELINES,
            PACKAGE_DIR / PROBLEM_BASELINES_COPY_RELATIVE,
        )
    else:
        raise PackageContractError("Sealed problem baselines drifted.")

    predecessor_cells = predecessor.get("cell_locks")
    if not isinstance(predecessor_cells, Mapping):
        raise PackageContractError("Predecessor source-lock cells are absent.")
    implementation = _implementation_source_inventory(REPO_ROOT)
    archive_relative_repo = (
        PACKAGE_DIR / AUTHORITY_ARCHIVE_RELATIVE
    ).relative_to(REPO_ROOT).as_posix()
    archive_sha = sha256_file(PACKAGE_DIR / AUTHORITY_ARCHIVE_RELATIVE)
    cells_by_identity = {
        (str(row["regime_id"]), int(row["nph"])): row
        for row in source_rows
    }
    target_cells = build_phase3_qiskit_mixed_horizon_cell_specs()
    derived_cells: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    for cell in target_cells:
        source_row = cells_by_identity.get((cell.regime_id, int(cell.nph)))
        predecessor_id = source_lock_id(cell.regime_id, cell.nph)
        raw_predecessor = predecessor_cells.get(predecessor_id)
        if not isinstance(source_row, Mapping) or not isinstance(
            raw_predecessor, Mapping
        ):
            raise PackageContractError(
                f"Missing page-7 predecessor for {cell.cell_id}."
            )
        predecessor_trace = raw_predecessor.get("resolver_trace")
        if not isinstance(predecessor_trace, Mapping):
            raise PackageContractError(
                f"Page-7 resolver trace is absent for {cell.cell_id}."
            )
        same_cutoff_ed = predecessor_trace.get("same_cutoff_ed_reference")
        if not isinstance(same_cutoff_ed, Mapping):
            raise PackageContractError(
                f"Same-cutoff ED reference is absent for {cell.cell_id}."
            )
        if int(same_cutoff_ed.get("nph", -1)) != int(cell.nph):
            raise PackageContractError(
                f"Same-cutoff ED reference drifted for {cell.cell_id}."
            )
        parent_authority = {
            "authority_kind": "exact_page7_resolved_protocol_bytes_v1",
            "archive": {
                "path": archive_relative_repo,
                "sha256": archive_sha,
            },
            "archive_member": {
                "path": source_row["path"],
                "sha256": source_row["sha256"],
            },
            "protocol_canonical_sha256": source_row["canonical_sha256"],
            "scientific_result_anchor_claimed": False,
        }
        changes = [
            {
                "id": "phase3_qiskit_selector_cost_scope",
                "field": "selector_compile_cost_scope",
                "from": "marrakesh_graph_span_all_phases_v1",
                "to": BACKEND_COMPILE_SCOPE,
                "classification": (
                    "explicit_user_requested_phase3_only_qiskit_"
                    "selector_cost_v1"
                ),
                "binding": (
                    "page7_parent_route_phase3_only_compile_cost_delta_v1"
                ),
            },
            {
                "id": "phase3_qiskit_exact_cell_selection",
                "field": "campaign_cell_selection",
                "from": (
                    f"page7__{cell.regime_id}__nph{cell.nph}__"
                    f"r{cell.horizon}"
                ),
                "to": cell.cell_id,
                "classification": (
                    "explicit_user_requested_six_cell_mixed_horizon_"
                    "candidate_v1"
                ),
                "binding": "phase3_qiskit_exact_cell_matrix_v1",
            },
        ]
        phase3_anchor = {
            "schema": "paper_i_ra_adapt_phase3_qiskit_source_anchor_v1",
            "source_algorithm_id": SOURCE_ALGORITHM_ID,
            "source_route_id": ROUTE_ID,
            "source_route_profile": SOURCE_ROUTE_PROFILE,
            "source_route_contract_sha256": (
                SOURCE_ROUTE_CONTRACT_SHA256
            ),
            "source_protocol_file_sha256": source_row["sha256"],
            "source_protocol_canonical_sha256": source_row[
                "canonical_sha256"
            ],
            "source_authority_manifest_sha256": authority_manifest["sha256"],
            "target_campaign_id": CAMPAIGN_ID,
            "target_bundle_id": BUNDLE_ID,
            "target_algorithm_id": ALGORITHM_ID,
            "regime_id": cell.regime_id,
            "nph": int(cell.nph),
            "source_horizon": int(cell.horizon or -1),
            "target_horizon": int(cell.horizon or -1),
            "scientific_result_anchor_claimed": False,
            "declared_delta_ids": [
                "phase3_qiskit_selector_cost_scope",
                "phase3_qiskit_exact_cell_selection",
            ],
        }
        trace = {
            "schema": (
                "paper_i_ra_adapt_phase3_qiskit_page7_parent_"
                "resolver_trace_v1"
            ),
            "source_map": authority_manifest["schema"],
            "regime_or_case": cell.regime_id,
            "method": ROUTE_ID,
            "source_json": str(source_row["path"]),
            "source_sha256_expected": str(source_row["sha256"]),
            "source_sha256_actual": str(source_row["sha256"]),
            "source_sha256_match": True,
            "settings_reused": {
                "settings": _settings_from_parent_protocol(
                    source_row["payload"],
                    cell_id=cell.cell_id,
                )
            },
            "settings_reused_sources": parent_authority,
            "settings_changed": changes,
            "same_cutoff_ed_reference": copy.deepcopy(
                dict(same_cutoff_ed)
            ),
            "phase3_qiskit_source_anchor": phase3_anchor,
            "page7_parent_protocol_authority": parent_authority,
            "status": "ok",
            "problems": [],
        }
        serialized_trace = canonical_json_bytes(trace)
        if b"9381198" in serialized_trace or b"all_six_r50" in serialized_trace:
            raise PackageContractError(
                f"Stale pre-page-7 authority leaked into {cell.cell_id}."
            )
        derived = {
            "regime_id": cell.regime_id,
            "nph": int(cell.nph),
            "route_id": cell.route_id,
            "archive": {
                "path": archive_relative_repo,
                "sha256": archive_sha,
            },
            "member": {
                "path": source_row["path"],
                "sha256": source_row["sha256"],
            },
            "resolver_trace": trace,
        }
        derived_cells[predecessor_id] = derived
        audit_rows.append(
            {
                "cell_id": cell.cell_id,
                "source_lock_id": predecessor_id,
                "source_protocol_member": source_row["path"],
                "source_protocol_file_sha256": source_row["sha256"],
                "source_protocol_canonical_sha256": source_row[
                    "canonical_sha256"
                ],
                "source_route_contract_sha256": (
                    SOURCE_ROUTE_CONTRACT_SHA256
                ),
                "source_horizon": int(cell.horizon or -1),
                "target_horizon": int(cell.horizon or -1),
                "declared_scientific_delta_ids": [
                    "phase3_qiskit_selector_cost_scope",
                    "phase3_qiskit_exact_cell_selection",
                ],
                "scientific_result_anchor_claimed": False,
                "status": "passed",
            }
        )

    source_locks = {
        "schema": predecessor["schema"],
        "global_sources": copy.deepcopy(predecessor["global_sources"]),
        "implementation_sources": implementation,
        "cell_locks": derived_cells,
    }
    receipt = digested(
        {
            "schema": (
                "paper_i_ra_adapt_global_singleton_phase3_qiskit_"
                "source_lock_derivation_v1"
            ),
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "source_authority_manifest_sha256": authority_manifest["sha256"],
            "predecessor_source_locks_file_sha256": (
                PREDECESSOR_SOURCE_LOCKS_FILE_SHA256
            ),
            "predecessor_source_locks_canonical_sha256": (
                PREDECESSOR_SOURCE_LOCKS_CANONICAL_SHA256
            ),
            "implementation_source_inventory_sha256": implementation["sha256"],
            "source_route_contract_sha256": (
                SOURCE_ROUTE_CONTRACT_SHA256
            ),
            "rows": audit_rows,
            "cell_count": len(audit_rows),
            "all_parent_protocol_bytes_verified": True,
            "all_page7_route_bindings_verified": True,
            "resolver_trace_reconstruction_policy": (
                "exact_page7_parent_protocol_only_no_historical_"
                "archive_authority_v1"
            ),
            "stale_cluster_9381198_authority_retained": False,
            "stale_core_fixed_horizon_delta_retained": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    _write_json(
        PACKAGE_DIR / "source_lock_derivation_receipt.json",
        receipt,
    )
    return source_locks, receipt


def _build_runtime_source_archive(
    normalized_source_locks: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    implementation = normalized_source_locks.get("implementation_sources")
    globals_raw = normalized_source_locks.get("global_sources")
    if not isinstance(implementation, Mapping) or not isinstance(
        globals_raw, Mapping
    ):
        raise PackageContractError("Normalized runtime source locks are absent.")
    files = implementation.get("files")
    if not isinstance(files, list):
        raise PackageContractError("Implementation source inventory is absent.")
    member_by_path: dict[str, dict[str, Any]] = {}
    for raw in files:
        if not isinstance(raw, Mapping):
            raise PackageContractError("Malformed implementation source row.")
        relative = str(raw.get("path", ""))
        source = REPO_ROOT / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or sha256_file(source) != raw.get("sha256")
        ):
            raise PackageContractError(
                f"Implementation source drifted: {relative}"
            )
        member_by_path[relative] = {
            "path": relative,
            "source_path": source,
            "sha256": raw["sha256"],
            "size_bytes": source.stat().st_size,
            "role": "implementation_source",
        }
    missing_route_sources = sorted(
        set(REQUIRED_PHASE3_QISKIT_SOURCE_PATHS).difference(member_by_path)
    )
    if missing_route_sources:
        raise PackageContractError(
            "Implementation closure omitted Phase-III-Qiskit source: "
            + ", ".join(missing_route_sources)
        )
    for role, raw in globals_raw.items():
        if not isinstance(raw, Mapping):
            raise PackageContractError(f"Malformed global source: {role}")
        relative = str(raw.get("path", ""))
        source = REPO_ROOT / relative
        if (
            not source.is_file()
            or source.is_symlink()
            or sha256_file(source) != raw.get("sha256")
        ):
            raise PackageContractError(f"Global source drifted: {relative}")
        prior = member_by_path.get(relative)
        row = {
            "path": relative,
            "source_path": source,
            "sha256": raw["sha256"],
            "size_bytes": source.stat().st_size,
            "role": f"global_source:{role}",
        }
        if prior is not None and (
            prior["sha256"], prior["size_bytes"]
        ) != (row["sha256"], row["size_bytes"]):
            raise PackageContractError(f"Source role hash conflict: {relative}")
        member_by_path[relative] = row
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.is_file() or requirements.is_symlink():
        raise PackageContractError("requirements.txt is missing or unsafe.")
    member_by_path["requirements.txt"] = {
        "path": "requirements.txt",
        "source_path": requirements,
        "sha256": sha256_file(requirements),
        "size_bytes": requirements.stat().st_size,
        "role": "runtime_dependency_lock",
    }
    rows = [member_by_path[key] for key in sorted(member_by_path)]
    archive = PACKAGE_DIR / "source/source_locked.tar.gz"
    _write_deterministic_archive(destination=archive, rows=rows)
    archive_binding = binding(archive, root=PACKAGE_DIR)
    manifest = digested(
        {
            "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA,
            "status": "passed",
            "archive": archive_binding,
            "members": [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "size_bytes": row["size_bytes"],
                    "role": row["role"],
                }
                for row in rows
            ],
            "member_count": len(rows),
            "implementation_source_inventory_sha256": implementation[
                "sha256"
            ],
            "global_source_paths": sorted(
                str(row["path"]) for row in globals_raw.values()
            ),
            "runtime_path_dependencies": ["requirements.txt"],
            "no_ambient_repo_imports": True,
        }
    )
    _write_json(PACKAGE_DIR / "source/source_archive_manifest.json", manifest)
    return archive_binding, manifest


def _validate_target_protocol(protocol: Any, *, horizon: int) -> None:
    route = protocol.route_contract
    execution = route.get("execution_settings")
    invariants = route.get("semantic_invariants")
    lineage = route.get("lineage_authority")
    if (
        protocol.algorithm_id != ALGORITHM_ID
        or protocol.candidate_representation != CANDIDATE_REPRESENTATION
        or protocol.request.adapter.adapter_id != CANDIDATE_ADAPTER_ID
        or int(protocol.horizon) != int(horizon)
        or route.get("route_profile") != TARGET_ROUTE_PROFILE
        or not isinstance(execution, Mapping)
        or execution.get("phase3_backend_cost_mode")
        != "marrakesh_graph_span_v1"
        or execution.get("phase3_backend_cost_scope") != BACKEND_COMPILE_SCOPE
        or execution.get("phase3_backend_name") != "FakeMarrakesh"
        or execution.get("phase3_backend_optimization_level") != 1
        or execution.get("phase3_backend_transpile_seed") != 7
        or execution.get("phase3_hardware_cost_normalization_mode")
        != "family_robust_symmetric_arctan_v1"
        or not isinstance(invariants, Mapping)
        or invariants.get("selector_compile_cost_policy")
        != SELECTOR_COMPILE_COST_POLICY
        or invariants.get("selector_compile_cost_phase_reuse")
        != SELECTOR_COMPILE_COST_PHASE_REUSE
        or invariants.get("selector_compile_cost_scope")
        != BACKEND_COMPILE_SCOPE
        or invariants.get("phase_iii_qiskit_independent_base_trial_layouts")
        is not True
        or invariants.get("phase_iii_qiskit_population_normalization_policy")
        != "family_robust_symmetric_arctan_v1"
        or invariants.get("phase_iii_qiskit_backend_fallback_allowed")
        is not False
        or invariants.get("phase_iii_qiskit_failure_policy") != "abort_run_v1"
        or not isinstance(lineage, Mapping)
        or lineage.get("parent_route_profile") != SOURCE_ROUTE_PROFILE
        or lineage.get("parent_contract_sha256")
        != SOURCE_ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError(
            f"Target Phase-III-Qiskit protocol drifted: {protocol.sha256}"
        )


def build() -> dict[str, Any]:
    if any(
        (PACKAGE_DIR / name).exists() or (PACKAGE_DIR / name).is_symlink()
        for name in GENERATED_PATHS
    ):
        raise FileExistsError(
            "Refusing to overwrite an existing Phase-III-Qiskit package."
        )
    for name in CONTROL_FILES:
        path = PACKAGE_DIR / name
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(f"Missing package control file: {name}")
    if any(
        path.name == "__pycache__" or path.suffix == ".pyc"
        for path in PACKAGE_DIR.rglob("*")
    ):
        raise PackageContractError("Unbound package bytecode is forbidden.")
    if PHASE3_QISKIT_BUNDLE_ID != BUNDLE_ID or (
        PHASE3_QISKIT_CAMPAIGN_ID != CAMPAIGN_ID
    ):
        raise PackageContractError("Typed campaign identity drifted.")
    if (
        PHASE3_QISKIT_PAGE7_PARENT_ROUTE_PROFILE != SOURCE_ROUTE_PROFILE
        or PHASE3_QISKIT_PAGE7_PARENT_CONTRACT_SHA256
        != SOURCE_ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError("Typed page-7 parent identity drifted.")

    source_packages = _verify_source_packages()
    source_rows, authority_manifest = _build_source_authority(source_packages)
    source_locks, derivation_receipt = _derive_source_locks(
        source_rows=source_rows,
        authority_manifest=authority_manifest,
    )
    baselines = load_json(
        REPO_ROOT / PROBLEM_BASELINES,
        label="sealed problem baselines",
    )
    materialized = materialize_phase3_qiskit_mixed_horizon_bundle(
        PACKAGE_DIR / "bundle_materialization",
        problem_resolver=support._problem_resolver_from(baselines),
        source_locks=source_locks,
        repository_state=_repository_state(),
        repo_root=REPO_ROOT,
        dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
        materialization_timestamp=support._utc_now(),
        verify_source_files=True,
    )
    if (
        materialized.bundle_id != BUNDLE_ID
        or materialized.materialization_status != "passed"
        or int(materialized.cell_count) != 6
        or materialized.bundle_path.resolve() != BUNDLE_ROOT.resolve()
    ):
        raise PackageContractError("Typed bundle materialization drifted.")

    bundle_manifest = load_json(
        BUNDLE_ROOT / "bundle_manifest.json",
        label="bundle manifest",
    )
    verify_self_digest(bundle_manifest, label="bundle manifest")
    normalized_locks = load_json(
        BUNDLE_ROOT / "source_locks.json",
        label="normalized source locks",
    )
    verify_self_digest(normalized_locks, label="normalized source locks")
    validation_report = load_json(
        BUNDLE_ROOT / "validation_report.json",
        label="bundle validation report",
    )
    verify_self_digest(validation_report, label="bundle validation report")
    expected_artifacts_manifest = load_json(
        BUNDLE_ROOT / "expected_artifacts.json",
        label="bundle expected artifacts",
    )
    verify_self_digest(
        expected_artifacts_manifest,
        label="bundle expected artifacts",
    )
    expected_artifact_cells = expected_artifacts_manifest.get("cells")
    if not isinstance(expected_artifact_cells, Mapping):
        raise PackageContractError("Bundle expected-artifact cells are absent.")
    expected_cells = build_phase3_qiskit_mixed_horizon_cell_specs()
    mixed_horizon_contract = bundle_manifest.get(
        "phase3_qiskit_mixed_horizon_contract"
    )
    if not isinstance(mixed_horizon_contract, Mapping):
        raise PackageContractError(
            "Bundle Phase-III-Qiskit mixed-horizon contract is absent."
        )
    if [cell.cell_id for cell in expected_cells] != list(
        mixed_horizon_contract.get("ordered_cell_ids", [])
    ):
        raise PackageContractError("Bundle ordered cell closure drifted.")

    protocol_bindings: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    route_digests: set[str] = set()
    for cell in expected_cells:
        protocol_path = BUNDLE_ROOT / "protocols" / f"{cell.cell_id}.json"
        protocol = load_validated_bundle_protocol(protocol_path)
        _validate_target_protocol(protocol, horizon=int(cell.horizon or -1))
        route_digest = str(protocol.route_contract.get("sha256", ""))
        if len(route_digest) != 64:
            raise PackageContractError(
                f"Target route digest is malformed: {cell.cell_id}."
            )
        route_digests.add(route_digest)
        protocol_binding = {
            "execution_id": cell.cell_id,
            **binding(protocol_path, root=PACKAGE_DIR, canonical=True),
        }
        protocol_bindings.append(protocol_binding)
        resources = copy.deepcopy(RESOURCE_ENVELOPES[int(cell.nph)])
        artifact_cell = expected_artifact_cells.get(cell.cell_id)
        expected_run_artifacts = (
            artifact_cell.get("expected_run_artifacts")
            if isinstance(artifact_cell, Mapping)
            else None
        )
        if not isinstance(expected_run_artifacts, Mapping):
            raise PackageContractError(
                f"Expected artifacts are absent for {cell.cell_id}."
            )
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "execution_id": cell.cell_id,
                "cell_id": cell.cell_id,
                "regime_id": cell.regime_id,
                "nph": int(cell.nph),
                "target_horizon": int(cell.horizon or -1),
                "algorithm_id": ALGORITHM_ID,
                "route_id": ROUTE_ID,
                "route_contract_sha256": route_digest,
                "source_route_contract_sha256": (
                    SOURCE_ROUTE_CONTRACT_SHA256
                ),
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
                "selector_compile_cost_scope": BACKEND_COMPILE_SCOPE,
                "protocol_path": protocol_binding["path"],
                "protocol_file_sha256": protocol_binding["sha256"],
                "protocol_sha256": protocol.sha256,
                "bundle_manifest_sha256": bundle_manifest["sha256"],
                "source_locks_sha256": normalized_locks["sha256"],
                "implementation_source_inventory_sha256": normalized_locks[
                    "implementation_sources"
                ]["sha256"],
                "expected_artifacts_manifest_sha256": (
                    expected_artifacts_manifest["sha256"]
                ),
                "expected_run_artifacts": copy.deepcopy(
                    dict(expected_run_artifacts)
                ),
                "resources": resources,
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
        job["job_path"] = f"jobs/{cell.cell_id}.json"
        job = digested(job)
        _write_json(PACKAGE_DIR / job["job_path"], job)
        jobs.append(job)
    if len(route_digests) != 1:
        raise PackageContractError(
            "Mixed-horizon protocols do not share one child route digest."
        )
    child_route_digest = next(iter(route_digests))

    archive_binding, source_archive_manifest = _build_runtime_source_archive(
        normalized_locks
    )
    audit = digested(
        {
            "schema": SOURCE_LOCK_AUDIT_SCHEMA,
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "source_authority_manifest_sha256": authority_manifest["sha256"],
            "source_lock_derivation_receipt_sha256": derivation_receipt[
                "sha256"
            ],
            "normalized_source_locks_sha256": normalized_locks["sha256"],
            "implementation_source_inventory_sha256": normalized_locks[
                "implementation_sources"
            ]["sha256"],
            "source_route_profile": SOURCE_ROUTE_PROFILE,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "child_route_profile": TARGET_ROUTE_PROFILE,
            "child_route_contract_sha256": child_route_digest,
            "declared_scientific_change": (
                "phase3_selector_cost_graph_span_to_qiskit_positive_clipped_"
                "marginal_transpile_v1"
            ),
            "unchanged_scientific_settings": [
                "global_guarded_singleton_phase_i_pool_v1",
                "identity_phase_ii_v1",
                "stationary_source_response_v1",
                "all_phase_resource_weighting_v1",
                "plateau_commutation_insertion_v2",
                "singleton_admission_v1",
                "powell_maxiter_200_seed_7_v1",
            ],
            "comparison_contract": {
                "weak_holstein": {
                    "candidate_horizon": 50,
                    "append_comparator_horizon": 50,
                },
                "strong_holstein": {
                    "candidate_horizon": 70,
                    "append_comparator_horizon": 70,
                },
                "primary": "final_same_cutoff_absolute_energy_error_v1",
                "secondary": [
                    "s_alg_at_final_horizon_v1",
                    "s_alg_at_first_common_error_crossing_v1",
                ],
            },
            "phase3_compile_contract": {
                "scope": BACKEND_COMPILE_SCOPE,
                "backend": "FakeMarrakesh",
                "optimization_level": 1,
                "transpile_seed": 7,
                "base_trial_layout_policy": (
                    "independent_unconstrained_full_transpiles_v1"
                ),
                "population_normalization": (
                    "family_robust_symmetric_arctan_v1"
                ),
                "excluded_from_s_alg": True,
            },
            "required_route_source_paths": list(
                REQUIRED_PHASE3_QISKIT_SOURCE_PATHS
            ),
            "pinned_execution_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
                "runtime_probe": (
                    "required_before_execution_or_submission_v1"
                ),
                "runtime_probe_status": "not_run_image_unavailable_locally",
            },
            "cell_count": len(jobs),
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "source_lock_audit.json", audit)

    queue_text = "".join(
        "\t".join(
            (
                job["execution_id"],
                job["job_path"],
                job["protocol_path"],
                job["sha256"],
                str(job["resources"]["request_cpus"]),
                str(job["resources"]["request_memory_mb"]),
                str(job["resources"]["request_disk_mb"]),
                str(job["resources"]["max_runtime_seconds"]),
            )
        )
        + "\n"
        for job in jobs
    )
    queue_path = PACKAGE_DIR / "queue.tsv"
    with queue_path.open("xb") as stream:
        stream.write(queue_text.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())

    execution_plan = digested(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "status": "passed_inert_six_cells",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "ordered_execution_ids": [job["execution_id"] for job in jobs],
            "row_count": len(jobs),
            "weak_holstein_horizon": 50,
            "strong_holstein_horizon": 70,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "child_route_contract_sha256": child_route_digest,
            "source_archive_sha256": archive_binding["sha256"],
            "source_locks_sha256": normalized_locks["sha256"],
            "implementation_source_inventory_sha256": normalized_locks[
                "implementation_sources"
            ]["sha256"],
            "submit_template_present": True,
            "submit_descriptor_present": False,
            "authorizations_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
            "remote_image_path": REMOTE_IMAGE_PATH,
            "remote_image_sha256": REMOTE_IMAGE_SHA256,
            "remote_image_runtime_probe_required": True,
            "remote_image_runtime_probe_status": (
                "not_run_image_unavailable_locally"
            ),
            "activation_policy": (
                "fresh_explicit_user_authority_plus_pinned_image_probe_v1"
            ),
            "activation_artifacts_present": False,
        }
    )
    _write_json(PACKAGE_DIR / "execution_plan.json", execution_plan)

    os.chmod(PACKAGE_DIR / "execute_authorized_job.sh", 0o755)
    controls = [
        binding(PACKAGE_DIR / name, root=PACKAGE_DIR) for name in CONTROL_FILES
    ]
    job_bindings = [
        binding(PACKAGE_DIR / job["job_path"], root=PACKAGE_DIR, canonical=True)
        for job in jobs
    ]
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
            "weak_holstein_horizon": 50,
            "strong_holstein_horizon": 70,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "child_route_contract_sha256": child_route_digest,
            "control_files": controls,
            "source_authority_manifest": binding(
                PACKAGE_DIR / AUTHORITY_MANIFEST_RELATIVE,
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "source_authority_archive": binding(
                PACKAGE_DIR / AUTHORITY_ARCHIVE_RELATIVE,
                root=PACKAGE_DIR,
            ),
            "source_lock_derivation_receipt": binding(
                PACKAGE_DIR / "source_lock_derivation_receipt.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_manifest": binding(
                BUNDLE_ROOT / "bundle_manifest.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_source_locks": binding(
                BUNDLE_ROOT / "source_locks.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_expected_artifacts": binding(
                BUNDLE_ROOT / "expected_artifacts.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_validation_report": binding(
                BUNDLE_ROOT / "validation_report.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "protocols": protocol_bindings,
            "jobs": job_bindings,
            "source_archive": archive_binding,
            "source_archive_manifest": binding(
                PACKAGE_DIR / "source/source_archive_manifest.json",
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
            "implementation_source_inventory_sha256": normalized_locks[
                "implementation_sources"
            ]["sha256"],
            "source_archive_manifest_sha256": source_archive_manifest[
                "sha256"
            ],
            "required_route_source_paths": list(
                REQUIRED_PHASE3_QISKIT_SOURCE_PATHS
            ),
            "remote_image_path": REMOTE_IMAGE_PATH,
            "remote_image_sha256": REMOTE_IMAGE_SHA256,
            "remote_image_runtime_probe_required": True,
            "remote_image_runtime_probe_status": (
                "not_run_image_unavailable_locally"
            ),
            "activation_policy": (
                "fresh_explicit_user_authority_plus_pinned_image_probe_v1"
            ),
            "activation_artifacts_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submit_template_present": True,
            "submit_descriptor_present": False,
            "authorizations_present": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / "package_manifest.json", manifest)

    from validate_package import validate_package

    validation = validate_package(deep=True)
    result = digested(
        {
            "schema": (
                "paper_i_ra_adapt_global_singleton_phase3_qiskit_"
                "local_package_build_receipt_v1"
            ),
            "status": "passed",
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "child_route_contract_sha256": child_route_digest,
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "source_archive_sha256": archive_binding["sha256"],
            "validation_sha256": validation["sha256"],
            "cell_count": len(jobs),
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    return result


def main() -> int:
    result = build()
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
